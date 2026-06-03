#include "../ds4.h"
#include "../ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
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
    const char *payload_out;
    int listen_port;
    int ctx_size;
    int gen_tokens;
    int forced_steps;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool debug;
} handoff_cfg;

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

typedef struct {
    int top1_saved;
    int top1_restored;
    int overlap;
    int top5_overlap;
    int nonfinite;
    float rms;
    float max_abs;
} logits_cmp;

typedef struct {
    int *tokens;
    int token_count;
    float *trace;
    int trace_steps;
    double elapsed_sec;
} token_trace_run;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void die_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE --listen-host IPV4 --listen-port N "
            "[--prompt-file FILE] [--system TEXT] [--ctx N] [--gen-tokens N] "
            "[--forced-steps N] [--prefill-chunk N] [--prefill-window N] "
            "[--activation-bits N] [--coordinator-layers A:B|A:output] "
            "[--worker-layers A:B|A:output] [--payload-out FILE] [--no-debug]\n",
            argv0);
    exit(2);
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase2-handoff: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "issue304-phase2-handoff: %s: %s\n", msg, strerror(errno));
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
    if (len != 0 && fread(buf, 1, (size_t)len, fp) != (size_t)len) {
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
            fprintf(stderr, "issue304-phase2-handoff: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase2-handoff: timed out waiting for route\n");
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

static void logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int i = 0; i < n; i++) {
        const float v = logits[i];
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int t = k - 1; t > j; t--) out[t] = out[t - 1];
                out[j] = i;
                break;
            }
        }
    }
}

static bool topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static logits_cmp compare_logits(const float *saved, const float *restored, int vocab) {
    int saved_top[20];
    int restored_top[20];
    logits_topk(saved, vocab, saved_top, 20);
    logits_topk(restored, vocab, restored_top, 20);

    int overlap = 0;
    int top5_overlap = 0;
    for (int i = 0; i < 20; i++) {
        if (saved_top[i] >= 0 && topk_contains(restored_top, 20, saved_top[i])) overlap++;
        if (i < 5 && saved_top[i] >= 0 && topk_contains(restored_top, 5, saved_top[i])) top5_overlap++;
    }

    int nonfinite = 0;
    double sumsq = 0.0;
    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(saved[i]) || !isfinite(restored[i])) {
            nonfinite++;
            continue;
        }
        float delta = restored[i] - saved[i];
        float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
    }

    logits_cmp out = {
        .top1_saved = saved_top[0],
        .top1_restored = restored_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .nonfinite = nonfinite,
        .rms = (float)sqrt(sumsq / (double)vocab),
        .max_abs = max_abs,
    };
    return out;
}

static bool load_payload_into_session(ds4_session *session,
                                      const ds4_session_payload_file *payload,
                                      char *err,
                                      size_t errlen) {
    FILE *fp = fopen(payload->path, "rb");
    if (!fp) {
        snprintf(err, errlen, "failed to open staged payload: %s", strerror(errno));
        return false;
    }
    const int rc = ds4_session_load_payload(session, fp, payload->bytes, err, errlen);
    fclose(fp);
    return rc == 0;
}

static bool write_payload_copy(const ds4_session_payload_file *payload,
                               const char *dst_path,
                               char *err,
                               size_t errlen) {
    FILE *fp = fopen(dst_path, "wb");
    if (!fp) {
        snprintf(err, errlen, "failed to open payload output: %s", strerror(errno));
        return false;
    }
    const int rc = ds4_session_write_staged_payload(payload, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        snprintf(err, errlen, "failed to close payload output: %s", strerror(errno));
        return false;
    }
    return rc == 0;
}

static bool write_binary_file(const char *path, const void *ptr, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool ok = fwrite(ptr, 1, bytes, fp) == bytes;
    ok = ok && fclose(fp) == 0;
    return ok;
}

static bool write_tokens_file(const char *path, const int *tokens, int count) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    if (fprintf(fp, "%d\n", count) < 0) {
        fclose(fp);
        return false;
    }
    for (int i = 0; i < count; i++) {
        if (fprintf(fp, "%d\n", tokens[i]) < 0) {
            fclose(fp);
            return false;
        }
    }
    return fclose(fp) == 0;
}

static char *sidecar_path(const char *base, const char *suffix) {
    if (!base || !suffix) return NULL;
    const size_t n = strlen(base) + strlen(suffix) + 1u;
    char *out = malloc(n);
    if (!out) return NULL;
    snprintf(out, n, "%s%s", base, suffix);
    return out;
}

static bool set_process_lock_file(const char *path, char *err, size_t errlen) {
    if (!path || !path[0]) {
        snprintf(err, errlen, "invalid lock file path");
        return false;
    }
    if (setenv("DS4_LOCK_FILE", path, 1) != 0) {
        snprintf(err, errlen, "failed to set DS4_LOCK_FILE: %s", strerror(errno));
        return false;
    }
    return true;
}

static bool capture_greedy_trace(ds4_session *session,
                                 int vocab,
                                 int steps,
                                 token_trace_run *out) {
    memset(out, 0, sizeof(*out));
    out->tokens = malloc((size_t)steps * sizeof(out->tokens[0]));
    out->trace = malloc((size_t)(steps + 1) * (size_t)vocab * sizeof(out->trace[0]));
    if (!out->tokens || !out->trace) return false;
    if (ds4_session_copy_logits(session, out->trace, vocab) != vocab) return false;

    char err[160];
    double t0 = now_sec();
    for (int i = 0; i < steps; i++) {
        const int token = ds4_session_argmax(session);
        out->tokens[i] = token;
        out->token_count++;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase2-handoff: greedy eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            return false;
        }
        if (ds4_session_copy_logits(session,
                                    out->trace + (size_t)(i + 1) * (size_t)vocab,
                                    vocab) != vocab) {
            return false;
        }
        out->trace_steps = i + 1;
    }
    out->elapsed_sec = now_sec() - t0;
    return true;
}

static bool capture_forced_trace(ds4_session *session,
                                 int vocab,
                                 const int *tokens,
                                 int steps,
                                 float *trace_out) {
    if (ds4_session_copy_logits(session, trace_out, vocab) != vocab) return false;
    char err[160];
    for (int i = 0; i < steps; i++) {
        if (ds4_session_eval(session, tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase2-handoff: forced eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            return false;
        }
        if (ds4_session_copy_logits(session,
                                    trace_out + (size_t)(i + 1) * (size_t)vocab,
                                    vocab) != vocab) {
            return false;
        }
    }
    return true;
}

static void free_token_trace_run(token_trace_run *run) {
    free(run->tokens);
    free(run->trace);
    memset(run, 0, sizeof(*run));
}

static void parse_args(handoff_cfg *cfg, int argc, char **argv) {
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
        } else if (!strcmp(argv[i], "--forced-steps") && i + 1 < argc) {
            cfg->forced_steps = atoi(argv[++i]);
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
        } else if (!strcmp(argv[i], "--payload-out") && i + 1 < argc) {
            cfg->payload_out = argv[++i];
        } else if (!strcmp(argv[i], "--no-debug")) {
            cfg->debug = false;
        } else {
            die_usage(argv[0]);
        }
    }
}

int main(int argc, char **argv) {
    handoff_cfg cfg = {
        .model_path = NULL,
        .prompt_file = "README.md",
        .system_prompt = "You are a helpful assistant",
        .listen_host = NULL,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
        .payload_out = NULL,
        .listen_port = 1234,
        .ctx_size = 16384,
        .gen_tokens = 16,
        .forced_steps = 8,
        .prefill_chunk = 256,
        .prefill_window = 0,
        .activation_bits = 32,
        .debug = true,
    };
    parse_args(&cfg, argc, argv);
    if (!cfg.model_path || !cfg.listen_host || cfg.listen_port <= 0 ||
        cfg.ctx_size <= 0 || cfg.gen_tokens <= 0 || cfg.forced_steps < 0) {
        die_usage(argv[0]);
    }
    if (cfg.forced_steps > cfg.gen_tokens) {
        die("--forced-steps must be <= --gen-tokens");
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

    ds4_engine *dist_engine = NULL;
    ds4_engine *local_engine = NULL;
    ds4_session *dist_session = NULL;
    ds4_session *local_session = NULL;
    ds4_session *trace_session = NULL;
    ds4_tokens prompt = {0};
    ds4_session_payload_file payload = {0};
    dsv4_header header;
    token_trace_run dist_run = {0};
    int *local_tokens = NULL;
    float *dist_logits = NULL;
    float *local_logits = NULL;
    float *local_trace = NULL;
    char err[256] = {0};
    bool stage_ok = false;
    bool payload_copy_ok = false;
    bool handoff_load_ok = false;
    bool trace_load_ok = false;
    bool greedy_match = false;
    bool forced_trace_match = false;
    int greedy_mismatch_step = -1;
    int greedy_dist_token = -1;
    int greedy_local_token = -1;
    int forced_first_bad_step = -1;
    int forced_token_before_step = -1;
    logits_cmp handoff_cmp = {0};
    logits_cmp forced_bad_cmp = {0};
    double prefill_sec = 0.0;
    double stage_sec = 0.0;
    double local_load_sec = 0.0;
    double local_decode_sec = 0.0;
    char payload_copy_err[256] = {0};
    char export_err[256] = {0};
    uint32_t route_hops = 0;
    bool output_on_coordinator = false;
    char route_summary[1024] = {0};
    int handoff_token_count = 0;
    uint64_t handoff_token_hash = 0;
    char *dist_logits_path = NULL;
    char *dist_tokens_path = NULL;
    char *dist_trace_path = NULL;

    ds4_engine_options dist_opt = {
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
    if (ds4_dist_prepare_engine_options(&dist_opt.distributed, &dist_opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase2-handoff: distributed options failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    if (ds4_engine_open(&dist_engine, &dist_opt) != 0 || !dist_engine) {
        die("failed to open distributed coordinator engine");
    }
    if (ds4_session_create(&dist_session, dist_engine, cfg.ctx_size) != 0 || !dist_session) {
        die("failed to create distributed coordinator session");
    }
    if (wait_route(dist_session, 60.0) != 1) goto fail;

    if (ds4_session_distributed_route_summary(dist_session,
                                              route_summary,
                                              sizeof(route_summary),
                                              &route_hops,
                                              &output_on_coordinator,
                                              err,
                                              sizeof(err)) != 1) {
        fprintf(stderr, "issue304-phase2-handoff: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }

    ds4_encode_chat_prompt(dist_engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) die("prompt tokenization failed");
    if (prompt.len >= cfg.ctx_size) die("prompt exceeds ctx");

    const int vocab = ds4_engine_vocab_size(dist_engine);
    dist_logits = malloc((size_t)vocab * sizeof(dist_logits[0]));
    local_logits = malloc((size_t)vocab * sizeof(local_logits[0]));
    if (!dist_logits || !local_logits) die("out of memory for logits");

    double t0 = now_sec();
    if (ds4_session_sync(dist_session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase2-handoff: distributed prefill failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    prefill_sec = now_sec() - t0;

    if (ds4_session_copy_logits(dist_session, dist_logits, vocab) != vocab) {
        die("failed to copy distributed handoff logits");
    }

    memset(&header, 0, sizeof(header));
    t0 = now_sec();
    if (ds4_session_stage_payload(dist_session, &payload, err, sizeof(err)) == 0) {
        stage_ok = true;
        if (!parse_dsv4_header(payload.path, &header)) {
            die("failed to parse staged DSV4 header");
        }
    } else {
        fprintf(stderr, "issue304-phase2-handoff: distributed payload stage failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }
    stage_sec = now_sec() - t0;

    if (cfg.payload_out) {
        payload_copy_ok = write_payload_copy(&payload,
                                             cfg.payload_out,
                                             payload_copy_err,
                                             sizeof(payload_copy_err));
        if (!payload_copy_ok) goto fail;
    }

    if (!capture_greedy_trace(dist_session, vocab, cfg.gen_tokens, &dist_run)) {
        goto fail;
    }

    if (cfg.payload_out) {
        dist_logits_path = sidecar_path(cfg.payload_out, ".logits.f32");
        dist_tokens_path = sidecar_path(cfg.payload_out, ".tokens.txt");
        dist_trace_path = sidecar_path(cfg.payload_out, ".trace.f32");
        if (!dist_logits_path || !dist_tokens_path || !dist_trace_path) {
            snprintf(export_err, sizeof(export_err), "out of memory creating sidecar paths");
            goto fail;
        }
        if (!write_binary_file(dist_logits_path,
                               dist_logits,
                               (size_t)vocab * sizeof(dist_logits[0]))) {
            snprintf(export_err, sizeof(export_err), "failed to write distributed logits sidecar");
            goto fail;
        }
        if (!write_tokens_file(dist_tokens_path, dist_run.tokens, dist_run.token_count)) {
            snprintf(export_err, sizeof(export_err), "failed to write distributed tokens sidecar");
            goto fail;
        }
        if (!write_binary_file(dist_trace_path,
                               dist_run.trace,
                               (size_t)(dist_run.token_count + 1) * (size_t)vocab * sizeof(dist_run.trace[0]))) {
            snprintf(export_err, sizeof(export_err), "failed to write distributed trace sidecar");
            goto fail;
        }
    }

    const ds4_tokens *timeline = ds4_session_tokens(dist_session);
    handoff_token_count = timeline ? timeline->len : 0;
    handoff_token_hash = timeline ? token_hash_prefix(timeline) : 0;

    ds4_session_free(dist_session);
    dist_session = NULL;
    ds4_engine_close(dist_engine);
    dist_engine = NULL;

    if (!set_process_lock_file("/tmp/ds4-phase2-local.lock", err, sizeof(err))) {
        fprintf(stderr, "issue304-phase2-handoff: %s\n", err);
        goto fail;
    }

    ds4_engine_options local_opt = {
        .model_path = cfg.model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
    };
    if (ds4_engine_open(&local_engine, &local_opt) != 0 || !local_engine) {
        die("failed to open local decode engine");
    }
    if (ds4_engine_vocab_size(local_engine) != vocab) {
        die("distributed and local vocab sizes differ");
    }

    if (ds4_session_create(&local_session, local_engine, cfg.ctx_size) != 0 || !local_session) {
        die("failed to create local decode session");
    }

    t0 = now_sec();
    handoff_load_ok = load_payload_into_session(local_session, &payload, err, sizeof(err));
    local_load_sec = now_sec() - t0;
    if (!handoff_load_ok) {
        fprintf(stderr, "issue304-phase2-handoff: local payload load failed: %s\n",
                err[0] ? err : "unknown error");
        goto fail;
    }

    if (ds4_session_copy_logits(local_session, local_logits, vocab) != vocab) {
        die("failed to copy local handoff logits");
    }
    handoff_cmp = compare_logits(dist_logits, local_logits, vocab);

    local_tokens = malloc((size_t)cfg.gen_tokens * sizeof(local_tokens[0]));
    if (!local_tokens) die("out of memory for local decode tokens");

    t0 = now_sec();
    for (int i = 0; i < cfg.gen_tokens; i++) {
        local_tokens[i] = ds4_session_argmax(local_session);
        if (ds4_session_eval(local_session, local_tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase2-handoff: local greedy eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            goto fail;
        }
    }
    local_decode_sec = now_sec() - t0;

    greedy_match = true;
    for (int i = 0; i < cfg.gen_tokens; i++) {
        if (dist_run.tokens[i] != local_tokens[i]) {
            greedy_match = false;
            greedy_mismatch_step = i;
            greedy_dist_token = dist_run.tokens[i];
            greedy_local_token = local_tokens[i];
            break;
        }
    }

    if (cfg.forced_steps > 0) {
        local_trace = malloc((size_t)(cfg.forced_steps + 1) * (size_t)vocab * sizeof(local_trace[0]));
        if (!local_trace) die("out of memory for local forced trace");

        if (ds4_session_create(&trace_session, local_engine, cfg.ctx_size) != 0 || !trace_session) {
            die("failed to create forced-trace session");
        }
        trace_load_ok = load_payload_into_session(trace_session, &payload, err, sizeof(err));
        if (!trace_load_ok) {
            fprintf(stderr, "issue304-phase2-handoff: forced-trace payload load failed: %s\n",
                    err[0] ? err : "unknown error");
            goto fail;
        }

        if (!capture_forced_trace(trace_session,
                                  vocab,
                                  dist_run.tokens,
                                  cfg.forced_steps,
                                  local_trace)) {
            goto fail;
        }

        forced_trace_match = true;
        for (int step = 0; step <= cfg.forced_steps; step++) {
            logits_cmp cmp = compare_logits(dist_run.trace + (size_t)step * (size_t)vocab,
                                            local_trace + (size_t)step * (size_t)vocab,
                                            vocab);
            if (cmp.nonfinite != 0 || cmp.top1_saved != cmp.top1_restored ||
                cmp.top5_overlap < 5 || cmp.overlap < 20 ||
                cmp.rms != 0.0f || cmp.max_abs != 0.0f) {
                forced_trace_match = false;
                forced_first_bad_step = step;
                forced_bad_cmp = cmp;
                forced_token_before_step = step > 0 ? dist_run.tokens[step - 1] : -1;
                break;
            }
        }
    }

    const uint64_t hash = handoff_token_hash;
    uint32_t hash_hi = 0;
    uint32_t hash_lo = 0;
    u64_to_halves(hash, &hash_hi, &hash_lo);

    printf("{\n");
    printf("  \"tool\":\"issue304_phase2_handoff\",\n");
    printf("  \"listen_host\":\"%s\",\n", cfg.listen_host);
    printf("  \"listen_port\":%d,\n", cfg.listen_port);
    printf("  \"model_path\":\"%s\",\n", cfg.model_path);
    printf("  \"prompt_file\":\"%s\",\n", cfg.prompt_file);
    printf("  \"ctx\":%d,\n", cfg.ctx_size);
    printf("  \"prompt_tokens\":%d,\n", prompt.len);
    printf("  \"gen_tokens\":%d,\n", cfg.gen_tokens);
    printf("  \"forced_steps\":%d,\n", cfg.forced_steps);
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
    printf("  \"stage_sec\":%.6f,\n", stage_sec);
    printf("  \"payload_bytes\":%llu,\n", (unsigned long long)payload.bytes);
    printf("  \"payload_copy_ok\":%s,\n", payload_copy_ok ? "true" : "false");
    printf("  \"payload_copy_path\":\"%s\",\n", cfg.payload_out ? cfg.payload_out : "");
    printf("  \"payload_copy_error\":\"%s\",\n", payload_copy_ok || !cfg.payload_out ? "" : payload_copy_err);
    printf("  \"local_load_ok\":%s,\n", handoff_load_ok ? "true" : "false");
    printf("  \"local_load_sec\":%.6f,\n", local_load_sec);
    printf("  \"distributed_decode_sec\":%.6f,\n", dist_run.elapsed_sec);
    printf("  \"distributed_decode_tok_per_sec\":%.2f,\n",
           dist_run.elapsed_sec > 0.0 ? (double)cfg.gen_tokens / dist_run.elapsed_sec : 0.0);
    printf("  \"local_decode_sec\":%.6f,\n", local_decode_sec);
    printf("  \"local_decode_tok_per_sec\":%.2f,\n",
           local_decode_sec > 0.0 ? (double)cfg.gen_tokens / local_decode_sec : 0.0);
    printf("  \"token_count\":%d,\n", handoff_token_count);
    printf("  \"token_hash\":\"0x%08x%08x\",\n", hash_hi, hash_lo);
    printf("  \"handoff\":{\n");
    printf("    \"top1_saved\":%d,\n", handoff_cmp.top1_saved);
    printf("    \"top1_restored\":%d,\n", handoff_cmp.top1_restored);
    printf("    \"top5_overlap\":%d,\n", handoff_cmp.top5_overlap);
    printf("    \"top20_overlap\":%d,\n", handoff_cmp.overlap);
    printf("    \"rms\":%.9g,\n", handoff_cmp.rms);
    printf("    \"max_abs\":%.9g,\n", handoff_cmp.max_abs);
    printf("    \"nonfinite\":%d,\n", handoff_cmp.nonfinite);
    printf("    \"match\":%s\n",
           (handoff_cmp.nonfinite == 0 &&
            handoff_cmp.top1_saved == handoff_cmp.top1_restored &&
            handoff_cmp.top5_overlap == 5 &&
            handoff_cmp.overlap == 20) ? "true" : "false");
    printf("  },\n");
    printf("  \"greedy\":{\n");
    printf("    \"match\":%s,\n", greedy_match ? "true" : "false");
    printf("    \"mismatch_step\":%d,\n", greedy_mismatch_step);
    printf("    \"distributed_token\":%d,\n", greedy_dist_token);
    printf("    \"local_token\":%d\n", greedy_local_token);
    printf("  },\n");
    printf("  \"forced_trace\":{\n");
    printf("    \"checked\":%s,\n", cfg.forced_steps > 0 ? "true" : "false");
    printf("    \"match\":%s,\n", forced_trace_match ? "true" : "false");
    printf("    \"first_bad_step\":%d,\n", forced_first_bad_step);
    printf("    \"forced_token_before_step\":%d,\n", forced_token_before_step);
    printf("    \"top1_ref\":%d,\n", forced_bad_cmp.top1_saved);
    printf("    \"top1_local\":%d,\n", forced_bad_cmp.top1_restored);
    printf("    \"top5_overlap\":%d,\n", forced_bad_cmp.top5_overlap);
    printf("    \"top20_overlap\":%d,\n", forced_bad_cmp.overlap);
    printf("    \"rms\":%.9g,\n", forced_bad_cmp.rms);
    printf("    \"max_abs\":%.9g,\n", forced_bad_cmp.max_abs);
    printf("    \"nonfinite\":%d\n", forced_bad_cmp.nonfinite);
    printf("  },\n");
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
    printf("}\n");

    free(dist_trace_path);
    free(dist_tokens_path);
    free(dist_logits_path);
    free(local_trace);
    free(local_logits);
    free(dist_logits);
    free(local_tokens);
    free_token_trace_run(&dist_run);
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(trace_session);
    ds4_session_free(local_session);
    ds4_session_free(dist_session);
    ds4_engine_close(local_engine);
    ds4_engine_close(dist_engine);
    free(prompt_text);
    return 0;

fail:
    if (export_err[0]) {
        fprintf(stderr, "issue304-phase2-handoff: %s\n", export_err);
    }
    free(dist_trace_path);
    free(dist_tokens_path);
    free(dist_logits_path);
    free(local_trace);
    free(local_logits);
    free(dist_logits);
    free(local_tokens);
    free_token_trace_run(&dist_run);
    ds4_session_payload_file_free(&payload);
    ds4_tokens_free(&prompt);
    ds4_session_free(trace_session);
    ds4_session_free(local_session);
    ds4_session_free(dist_session);
    ds4_engine_close(local_engine);
    ds4_engine_close(dist_engine);
    free(prompt_text);
    return 1;
}
