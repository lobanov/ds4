#include "../ds4.h"
#include "../ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
    const char *model_path;
    const char *prompt_file;
    const char *system_prompt;
    const char *listen_host;
    const char *coordinator_layers;
    const char *worker_layers;
    int listen_port;
    int ctx_size;
    int gen_tokens;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool debug;
} dgx_cfg;

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

static void die_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE --listen-host IPV4 --listen-port N "
            "[--prompt-file FILE] [--system TEXT] [--ctx N] [--gen-tokens N] "
            "[--prefill-chunk N] [--prefill-window N] [--activation-bits N] "
            "[--coordinator-layers A:B|A:output] [--worker-layers A:B|A:output] [--no-debug]\n",
            argv0);
    exit(2);
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase0-dgx: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "issue304-phase0-dgx: %s: %s\n", msg, strerror(errno));
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

static void u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static uint64_t token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t token_hash_prefix(const ds4_tokens *tokens) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < tokens->len; i++) h = token_hash_update(h, tokens->v[i]);
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

static int wait_route(ds4_session *session, double timeout_sec) {
    double deadline = now_sec() + timeout_sec;
    while (now_sec() < deadline) {
        char err[256] = {0};
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready == 1) return 1;
        if (ready < 0) {
            fprintf(stderr, "issue304-phase0-dgx: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase0-dgx: timed out waiting for route\n");
    return 0;
}

static bool parse_layer_spec(const char *spec, ds4_distributed_layers *out) {
    if (!spec || !out) return false;
    const char *colon = strchr(spec, ':');
    if (!colon || colon == spec || colon[1] == '\0') return false;

    errno = 0;
    char *end = NULL;
    unsigned long start = strtoul(spec, &end, 10);
    if (errno != 0 || end != colon || start > UINT32_MAX) return false;

    ds4_distributed_layers layers = {0};
    layers.start = (uint32_t)start;
    layers.set = true;
    if (strcmp(colon + 1, "output") == 0) {
        layers.has_output = true;
        layers.end = 0;
    } else {
        errno = 0;
        unsigned long layer_end = strtoul(colon + 1, &end, 10);
        if (errno != 0 || *end != '\0' || layer_end > UINT32_MAX) return false;
        layers.end = (uint32_t)layer_end;
    }
    *out = layers;
    return true;
}

static void parse_args(dgx_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--system") && i + 1 < argc) {
            cfg->system_prompt = argv[++i];
        } else if (!strcmp(argv[i], "--listen-host") && i + 1 < argc) {
            cfg->listen_host = argv[++i];
        } else if (!strcmp(argv[i], "--listen-port") && i + 1 < argc) {
            cfg->listen_port = atoi(argv[++i]);
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
        } else if (!strcmp(argv[i], "--coordinator-layers") && i + 1 < argc) {
            cfg->coordinator_layers = argv[++i];
        } else if (!strcmp(argv[i], "--worker-layers") && i + 1 < argc) {
            cfg->worker_layers = argv[++i];
        } else if (!strcmp(argv[i], "--no-debug")) {
            cfg->debug = false;
        } else {
            die_usage(argv[0]);
        }
    }
}

int main(int argc, char **argv) {
    dgx_cfg cfg = {
        .model_path = NULL,
        .prompt_file = "README.md",
        .system_prompt = "You are a helpful assistant",
        .listen_host = NULL,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
        .listen_port = 1234,
        .ctx_size = 8192,
        .gen_tokens = 16,
        .prefill_chunk = 256,
        .prefill_window = 0,
        .activation_bits = 32,
        .debug = true,
    };
    parse_args(&cfg, argc, argv);
    if (!cfg.model_path || !cfg.listen_host || cfg.listen_port <= 0 || cfg.ctx_size <= 0 || cfg.gen_tokens <= 0) {
        die_usage(argv[0]);
    }

    struct in_addr addr;
    if (inet_pton(AF_INET, cfg.listen_host, &addr) != 1) {
        die("listen host must be an IPv4 address reachable from the worker");
    }

    ds4_distributed_layers coordinator_layers = {0};
    if (!parse_layer_spec(cfg.coordinator_layers, &coordinator_layers)) {
        die("invalid coordinator layer spec");
    }

    char *prompt_text = read_file(cfg.prompt_file);
    if (!prompt_text) die_errno("read prompt file");

    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0};
    ds4_session_payload_file payload = {0};
    dsv4_header header;
    bool stage_ok = false;
    char stage_err[256] = {0};

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
            .layers = coordinator_layers,
            .listen_host = cfg.listen_host,
            .listen_port = cfg.listen_port,
            .prefill_chunk = cfg.prefill_chunk,
            .prefill_window = cfg.prefill_window,
            .activation_bits = cfg.activation_bits,
            .debug = cfg.debug,
        },
    };

    char err[256] = {0};
    if (ds4_dist_prepare_engine_options(&opt.distributed, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase0-dgx: distributed options failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        die("failed to open coordinator engine");
    }
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0 || !session) {
        die("failed to create coordinator session");
    }
    if (wait_route(session, 60.0) != 1) goto fail;

    char route_summary[1024] = {0};
    uint32_t route_hops = 0;
    bool output_on_coordinator = false;
    if (ds4_session_distributed_route_summary(session,
                                              route_summary,
                                              sizeof(route_summary),
                                              &route_hops,
                                              &output_on_coordinator,
                                              err,
                                              sizeof(err)) != 1) {
        fprintf(stderr, "issue304-phase0-dgx: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }

    ds4_encode_chat_prompt(engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) die("prompt tokenization failed");
    if (prompt.len >= cfg.ctx_size) die("prompt exceeds ctx");

    double t0 = now_sec();
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase0-dgx: distributed prefill failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    double t1 = now_sec();
    double prefill_sec = t1 - t0;

    memset(&header, 0, sizeof(header));
    double t2 = now_sec();
    if (ds4_session_stage_payload(session, &payload, err, sizeof(err)) == 0) {
        stage_ok = true;
        if (!parse_dsv4_header(payload.path, &header)) {
            die("failed to parse staged DSV4 header");
        }
    } else {
        snprintf(stage_err, sizeof(stage_err), "%s", err[0] ? err : "unknown error");
        fprintf(stderr, "issue304-phase0-dgx: distributed payload stage failed: %s\n", stage_err);
    }
    double t3 = now_sec();

    const ds4_tokens *timeline = ds4_session_tokens(session);
    const uint64_t hash = token_hash_prefix(timeline);
    uint32_t hash_hi = 0;
    uint32_t hash_lo = 0;
    u64_to_halves(hash, &hash_hi, &hash_lo);

    double t4 = now_sec();
    for (int i = 0; i < cfg.gen_tokens; i++) {
        int token = ds4_session_argmax(session);
        if (i + 1 < cfg.gen_tokens &&
            ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase0-dgx: distributed decode failed: %s\n",
                    err[0] ? err : "unknown error");
            goto fail;
        }
    }
    double t5 = now_sec();

    printf("{\n");
    printf("  \"tool\":\"issue304_phase0_dgx\",\n");
    printf("  \"listen_host\":\"%s\",\n", cfg.listen_host);
    printf("  \"listen_port\":%d,\n", cfg.listen_port);
    printf("  \"model_path\":\"%s\",\n", cfg.model_path);
    printf("  \"prompt_file\":\"%s\",\n", cfg.prompt_file);
    printf("  \"ctx\":%d,\n", cfg.ctx_size);
    printf("  \"prompt_tokens\":%d,\n", prompt.len);
    printf("  \"gen_tokens\":%d,\n", cfg.gen_tokens);
    printf("  \"prefill_chunk\":%u,\n", cfg.prefill_chunk);
    printf("  \"prefill_window\":%u,\n", cfg.prefill_window);
    printf("  \"activation_bits\":%u,\n", cfg.activation_bits);
    printf("  \"debug\":%s,\n", cfg.debug ? "true" : "false");
    printf("  \"coordinator_layers\":\"%s\",\n", cfg.coordinator_layers);
    printf("  \"expected_worker_layers\":\"%s\",\n", cfg.worker_layers);
    printf("  \"route_hops\":%u,\n", route_hops);
    printf("  \"route_summary\":\"%s\",\n", route_summary);
    printf("  \"output_owner\":\"%s\",\n", output_on_coordinator ? "coordinator" : "worker");
    printf("  \"prefill_sec\":%.6f,\n", prefill_sec);
    printf("  \"prefill_tok_per_sec\":%.2f,\n", prefill_sec > 0.0 ? (double)prompt.len / prefill_sec : 0.0);
    printf("  \"stage_ok\":%s,\n", stage_ok ? "true" : "false");
    printf("  \"stage_sec\":%.6f,\n", t3 - t2);
    printf("  \"payload_bytes\":%llu,\n", (unsigned long long)payload.bytes);
    printf("  \"stage_error\":\"%s\",\n", stage_ok ? "" : stage_err);
    printf("  \"decode_sec\":%.6f,\n", t5 - t4);
    printf("  \"decode_tok_per_sec\":%.2f,\n", (t5 - t4) > 0.0 ? (double)cfg.gen_tokens / (t5 - t4) : 0.0);
    printf("  \"token_count\":%d,\n", timeline ? timeline->len : 0);
    printf("  \"token_hash\":\"0x%08x%08x\",\n", hash_hi, hash_lo);
    if (stage_ok) {
        printf("  \"dsv4\":{\n");
        printf("    \"ctx_size\":%u,\n", header.ctx_size);
        printf("    \"prefill_cap\":%u,\n", header.prefill_cap);
        printf("    \"raw_cap\":%u,\n", header.raw_cap);
        printf("    \"raw_window\":%u,\n", header.raw_window);
        printf("    \"comp_cap\":%u,\n", header.comp_cap);
        printf("    \"saved_tokens\":%u,\n", header.saved_tokens);
        printf("    \"n_layer\":%u,\n", header.n_layer);
        printf("    \"head_dim\":%u,\n", header.head_dim);
        printf("    \"indexer_head_dim\":%u,\n", header.indexer_head_dim);
        printf("    \"vocab\":%u,\n", header.vocab);
        printf("    \"raw_live\":%u\n", header.raw_live);
        printf("  }\n");
    } else {
        printf("  \"dsv4\":null\n");
    }
    printf("}\n");

    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(prompt_text);
    return 0;

fail:
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(prompt_text);
    return 1;
}
