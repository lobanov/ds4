#include "../ds4.h"
#include "../ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

typedef struct {
    const char *model_path;
    const char *prompt_file;
    const char *worker_bin;
    const char *host;
    int port;
    int ctx_size;
    int gen_tokens;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool debug;
} local_cfg;

typedef struct {
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t comp_cap;
    uint32_t saved_tokens;
    uint32_t n_layer;
    uint32_t head_dim;
    uint32_t indexer_head_dim;
    uint32_t vocab;
    uint32_t raw_live;
} dsv4_header;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase0-local: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "issue304-phase0-local: %s: %s\n", msg, strerror(errno));
    exit(1);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    return buf;
}

static int pick_port(const char *host) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die_errno("socket");

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(0);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        die("invalid host for port probe");
    }
    if (bind(fd, (const struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        die_errno("bind port probe");
    }

    socklen_t len = (socklen_t)sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &len) != 0) {
        close(fd);
        die_errno("getsockname");
    }
    int port = (int)ntohs(addr.sin_port);
    close(fd);
    return port;
}

static void u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t dist_token_hash_prefix(const ds4_tokens *tokens) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < tokens->len; i++) h = dist_token_hash_update(h, tokens->v[i]);
    return h;
}

static bool read_u32_le(FILE *fp, uint32_t *out) {
    uint8_t buf[4];
    if (fread(buf, 1, sizeof(buf), fp) != sizeof(buf)) return false;
    *out = (uint32_t)buf[0] |
           ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) |
           ((uint32_t)buf[3] << 24);
    return true;
}

static bool parse_dsv4_header(const char *path, dsv4_header *out) {
    memset(out, 0, sizeof(*out));
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (!read_u32_le(fp, &h[i])) {
            fclose(fp);
            return false;
        }
    }
    fclose(fp);

    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        return false;
    }

    out->ctx_size = h[2];
    out->prefill_cap = h[3];
    out->raw_cap = h[4];
    out->raw_window = h[5];
    out->comp_cap = h[6];
    out->saved_tokens = h[7];
    out->n_layer = h[8];
    out->head_dim = h[9];
    out->indexer_head_dim = h[10];
    out->vocab = h[11];
    out->raw_live = h[12];
    return true;
}

static void kill_worker(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        int status = 0;
        pid_t got = waitpid(pid, &status, WNOHANG);
        if (got == pid) return;
        usleep(100000);
    }
    kill(pid, SIGKILL);
    (void)waitpid(pid, NULL, 0);
}

static pid_t spawn_worker(const local_cfg *cfg) {
    char port[16];
    snprintf(port, sizeof(port), "%d", cfg->port);

    char *const argv[] = {
        (char *)cfg->worker_bin,
        "-m", (char *)cfg->model_path,
        "--role", "worker",
        "--layers", "22:output",
        "--coordinator", (char *)cfg->host, port,
        NULL,
    };

    pid_t pid = 0;
    int rc = posix_spawn(&pid, cfg->worker_bin, NULL, NULL, argv, environ);
    if (rc != 0) {
        errno = rc;
        die_errno("posix_spawn worker");
    }
    return pid;
}

static int wait_route(ds4_session *session, pid_t worker_pid, double timeout_sec) {
    double deadline = now_sec() + timeout_sec;
    while (now_sec() < deadline) {
        int status = 0;
        pid_t got = waitpid(worker_pid, &status, WNOHANG);
        if (got == worker_pid) {
            if (WIFEXITED(status)) {
                fprintf(stderr,
                        "issue304-phase0-local: worker exited before route formation with status %d\n",
                        WEXITSTATUS(status));
            } else if (WIFSIGNALED(status)) {
                fprintf(stderr,
                        "issue304-phase0-local: worker exited before route formation from signal %d\n",
                        WTERMSIG(status));
            } else {
                fprintf(stderr, "issue304-phase0-local: worker exited before route formation\n");
            }
            return -1;
        }
        char err[256] = {0};
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready == 1) return 1;
        if (ready < 0) {
            fprintf(stderr, "issue304-phase0-local: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase0-local: timed out waiting for route\n");
    return 0;
}

static void parse_args(local_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--worker-bin") && i + 1 < argc) {
            cfg->worker_bin = argv[++i];
        } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            cfg->host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            cfg->port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            cfg->ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gen-tokens") && i + 1 < argc) {
            cfg->gen_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) {
            cfg->prefill_chunk = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--prefill-window") && i + 1 < argc) {
            cfg->prefill_window = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--activation-bits") && i + 1 < argc) {
            cfg->activation_bits = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--no-debug")) {
            cfg->debug = false;
        } else {
            fprintf(stderr,
                    "usage: %s [--model FILE] [--prompt-file FILE] [--worker-bin FILE] [--host IPV4] [--port N] [--ctx N] [--gen-tokens N] [--prefill-chunk N] [--prefill-window N] [--activation-bits N] [--no-debug]\n",
                    argv[0]);
            exit(2);
        }
    }
}

int main(int argc, char **argv) {
    local_cfg cfg = {
        .model_path = "ds4flash.gguf",
        .prompt_file = "README.md",
        .worker_bin = "./ds4",
        .host = "127.0.0.1",
        .port = 0,
        .ctx_size = 8192,
        .gen_tokens = 16,
        .prefill_chunk = 256,
        .prefill_window = 0,
        .activation_bits = 32,
        .debug = true,
    };
    parse_args(&cfg, argc, argv);
    if (cfg.port == 0) cfg.port = pick_port(cfg.host);

    pid_t worker_pid = 0;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_session_payload_file payload = {0};
    char *prompt_text = NULL;

    prompt_text = read_file(cfg.prompt_file);
    if (!prompt_text) die_errno("read prompt file");

    worker_pid = spawn_worker(&cfg);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .distributed = {
            .role = DS4_DISTRIBUTED_COORDINATOR,
            .layers = {
                .start = 0,
                .end = 21,
                .has_output = false,
                .set = true,
            },
            .listen_host = cfg.host,
            .listen_port = cfg.port,
            .prefill_chunk = cfg.prefill_chunk,
            .prefill_window = cfg.prefill_window,
            .activation_bits = cfg.activation_bits,
            .debug = cfg.debug,
        },
    };

    char err[256] = {0};
    if (ds4_dist_prepare_engine_options(&opt.distributed, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase0-local: distributed options failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        die("failed to open coordinator engine");
    }
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0 || !session) {
        die("failed to create coordinator session");
    }
    if (wait_route(session, worker_pid, 30.0) != 1) goto fail;

    ds4_encode_chat_prompt(engine, "You are a helpful assistant", prompt_text, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) die("prompt tokenization failed");
    if (prompt.len >= cfg.ctx_size) die("prompt exceeds ctx");
    const uint32_t chunks = (uint32_t)((prompt.len + (int)cfg.prefill_chunk - 1) / (int)cfg.prefill_chunk);
    if (chunks < 2u) die("prompt does not exercise multi-chunk distributed prefill; use a longer prompt or smaller --prefill-chunk");

    printf("PHASE0_LOCAL_CONFIG host=%s port=%d model=%s prompt_file=%s ctx=%d prefill_chunk=%u prefill_window=%u activation_bits=%u gen_tokens=%d worker_layers=22:output coordinator_layers=0:21\n",
           cfg.host, cfg.port, cfg.model_path, cfg.prompt_file, cfg.ctx_size,
           cfg.prefill_chunk, cfg.prefill_window, cfg.activation_bits, cfg.gen_tokens);
    printf("PHASE0_LOCAL_PROMPT tokens=%d chunks=%u\n", prompt.len, chunks);

    double t0 = now_sec();
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase0-local: distributed prefill failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    double t1 = now_sec();
    double prefill_sec = t1 - t0;

    double t2 = now_sec();
    if (ds4_session_stage_payload(session, &payload, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase0-local: distributed payload stage failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    double t3 = now_sec();

    dsv4_header header;
    if (!parse_dsv4_header(payload.path, &header)) {
        die("failed to parse staged DSV4 header");
    }

    const ds4_tokens *timeline = ds4_session_tokens(session);
    const uint64_t token_hash = dist_token_hash_prefix(timeline);
    uint32_t hash_hi = 0, hash_lo = 0;
    u64_to_halves(token_hash, &hash_hi, &hash_lo);

    double t4 = now_sec();
    for (int i = 0; i < cfg.gen_tokens; i++) {
        int token = ds4_session_argmax(session);
        if (i + 1 < cfg.gen_tokens &&
            ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase0-local: distributed decode failed: %s\n",
                    err[0] ? err : "unknown error");
            goto fail;
        }
    }
    double t5 = now_sec();

    printf("PHASE0_LOCAL_PREFILL tokens=%d sec=%.6f tok_per_sec=%.2f\n",
           prompt.len, prefill_sec, (double)prompt.len / prefill_sec);
    printf("PHASE0_LOCAL_STAGE bytes=%llu sec=%.6f mib=%.2f\n",
           (unsigned long long)payload.bytes,
           t3 - t2,
           (double)payload.bytes / (1024.0 * 1024.0));
    printf("PHASE0_LOCAL_DECODE tokens=%d sec=%.6f tok_per_sec=%.2f\n",
           cfg.gen_tokens, t5 - t4, (double)cfg.gen_tokens / (t5 - t4));
    printf("PHASE0_LOCAL_ROUTE coordinator=0:21 worker=22:output output_owner=worker debug=%d\n",
           cfg.debug ? 1 : 0);
    printf("PHASE0_LOCAL_HASH token_count=%d token_hash=0x%08x%08x\n",
           timeline ? timeline->len : 0, hash_hi, hash_lo);
    printf("PHASE0_LOCAL_DSV4 ctx_size=%u prefill_cap=%u raw_cap=%u raw_window=%u comp_cap=%u saved_tokens=%u n_layer=%u head_dim=%u indexer_head_dim=%u vocab=%u raw_live=%u\n",
           header.ctx_size, header.prefill_cap, header.raw_cap, header.raw_window,
           header.comp_cap, header.saved_tokens, header.n_layer, header.head_dim,
           header.indexer_head_dim, header.vocab, header.raw_live);

    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(prompt_text);
    kill_worker(worker_pid);
    return 0;

fail:
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(prompt_text);
    kill_worker(worker_pid);
    return 1;
}
