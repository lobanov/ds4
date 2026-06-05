#include "../ds4.h"
#include "../ds4_distributed.h"

#include <ctype.h>
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
    int listen_port;
    int ctx_size;
    int gen_tokens;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool debug;
} diag_cfg;

typedef struct {
    int *tokens;
    int token_count;
    float *trace;
    int trace_steps;
    double elapsed_sec;
} token_trace_run;

typedef struct {
    int top1_ref;
    int top1_cand;
    int top5_overlap;
    int top10_overlap;
    int top20_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
} parity_metrics;

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
    fprintf(stderr, "issue304-phase4-diagnose: %s\n", msg);
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

static int wait_route(ds4_session *session, double timeout_sec) {
    double deadline = now_sec() + timeout_sec;
    while (now_sec() < deadline) {
        char err[256] = {0};
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready == 1) return 1;
        if (ready < 0) {
            fprintf(stderr, "issue304-phase4-diagnose: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase4-diagnose: timed out waiting for route\n");
    return 0;
}

static void free_token_trace_run(token_trace_run *run) {
    free(run->tokens);
    free(run->trace);
    memset(run, 0, sizeof(*run));
}

static void parse_args(diag_cfg *cfg, int argc, char **argv) {
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

    if (!cfg->model_path || !cfg->listen_host || cfg->listen_port <= 0) {
        die_usage(argv[0]);
    }
}

static bool open_distributed_session(const diag_cfg *cfg,
                                     ds4_engine *engine,
                                     ds4_session **session_out,
                                     char *route_summary,
                                     size_t route_summary_len,
                                     uint32_t *route_hops,
                                     bool *output_on_coordinator) {
    *session_out = NULL;
    if (ds4_session_create(session_out, engine, cfg->ctx_size) != 0 || !*session_out) {
        fprintf(stderr, "issue304-phase4-diagnose: failed to create distributed session\n");
        return false;
    }
    if (wait_route(*session_out, 60.0) != 1) return false;

    char err[256] = {0};
    if (ds4_session_distributed_route_summary(*session_out,
                                              route_summary,
                                              route_summary_len,
                                              route_hops,
                                              output_on_coordinator,
                                              err,
                                              sizeof(err)) != 1) {
        fprintf(stderr, "issue304-phase4-diagnose: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
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

static parity_metrics compare_logits(const float *ref, const float *cand, int vocab) {
    int ref_top[64];
    int cand_top[64];
    logits_topk(ref, vocab, ref_top, 64);
    logits_topk(cand, vocab, cand_top, 64);

    int top5_overlap = 0;
    int top10_overlap = 0;
    int top20_overlap = 0;
    for (int i = 0; i < 20; i++) {
        if (ref_top[i] < 0) continue;
        if (i < 5 && topk_contains(cand_top, 5, ref_top[i])) top5_overlap++;
        if (i < 10 && topk_contains(cand_top, 10, ref_top[i])) top10_overlap++;
        if (topk_contains(cand_top, 20, ref_top[i])) top20_overlap++;
    }

    int nonfinite = 0;
    double sumsq = 0.0;
    float max_abs = 0.0f;
    float top20_max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(ref[i]) || !isfinite(cand[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand[i] - ref[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
    }
    for (int i = 0; i < 20; i++) {
        const int id = ref_top[i];
        if (id < 0) continue;
        const float abs_delta = fabsf(cand[id] - ref[id]);
        if (abs_delta > top20_max_abs) top20_max_abs = abs_delta;
    }

    parity_metrics out = {
        .top1_ref = ref_top[0],
        .top1_cand = cand_top[0],
        .top5_overlap = top5_overlap,
        .top10_overlap = top10_overlap,
        .top20_overlap = top20_overlap,
        .nonfinite = nonfinite,
        .rms = (float)sqrt(sumsq / (double)vocab),
        .max_abs = max_abs,
        .top20_max_abs = top20_max_abs,
    };
    return out;
}

static char *token_piece_text(ds4_engine *engine, int token) {
    size_t len = 0;
    char *piece = ds4_token_text(engine, token, &len);
    if (!piece) return strdup("");
    char *out = malloc(len + 1u);
    if (!out) {
        free(piece);
        return NULL;
    }
    memcpy(out, piece, len);
    out[len] = '\0';
    free(piece);
    return out;
}

static char *tokens_render_text(ds4_engine *engine, const int *tokens, int n) {
    size_t cap = 128;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';
    for (int i = 0; i < n; i++) {
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, tokens[i], &piece_len);
        if (!piece) continue;
        if (len + piece_len + 1u > cap) {
            size_t need = len + piece_len + 1u;
            size_t next = cap;
            while (next < need) next *= 2u;
            char *grown = realloc(buf, next);
            if (!grown) {
                free(piece);
                free(buf);
                return NULL;
            }
            buf = grown;
            cap = next;
        }
        memcpy(buf + len, piece, piece_len);
        len += piece_len;
        buf[len] = '\0';
        free(piece);
    }
    return buf;
}

static void print_json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", stdout); break;
            case '"': fputs("\\\"", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (c < 0x20u) {
                    printf("\\u%04x", c);
                } else {
                    putchar((char)c);
                }
                break;
        }
    }
    putchar('"');
}

static void print_token_array(const int *tokens, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        if (i != 0) putchar(',');
        printf("%d", tokens[i]);
    }
    putchar(']');
}

static void print_topk_json(ds4_engine *engine, const float *logits, int vocab, int k) {
    int top[16];
    if (k > (int)(sizeof(top) / sizeof(top[0]))) k = (int)(sizeof(top) / sizeof(top[0]));
    logits_topk(logits, vocab, top, k);
    putchar('[');
    for (int i = 0; i < k; i++) {
        char *piece = token_piece_text(engine, top[i]);
        if (i != 0) putchar(',');
        printf("{\"id\":%d,\"logit\":%.6f,\"text\":", top[i], logits[top[i]]);
        print_json_string(piece ? piece : "");
        printf("}");
        free(piece);
    }
    putchar(']');
}

static bool capture_reference_trace(ds4_session *session,
                                    int vocab,
                                    int steps,
                                    token_trace_run *out) {
    memset(out, 0, sizeof(*out));
    out->tokens = malloc((size_t)steps * sizeof(out->tokens[0]));
    out->trace = malloc((size_t)steps * (size_t)vocab * sizeof(out->trace[0]));
    if (!out->tokens || !out->trace) return false;

    char err[160] = {0};
    double t0 = now_sec();
    for (int i = 0; i < steps; i++) {
        const int token = ds4_session_argmax(session);
        out->tokens[i] = token;
        out->token_count++;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase4-diagnose: distributed eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            return false;
        }
        if (ds4_session_copy_logits(session,
                                    out->trace + (size_t)i * (size_t)vocab,
                                    vocab) != vocab) {
            fprintf(stderr, "issue304-phase4-diagnose: failed to copy distributed logits at step %d\n",
                    i);
            return false;
        }
        out->trace_steps = i + 1;
    }
    out->elapsed_sec = now_sec() - t0;
    return true;
}

int main(int argc, char **argv) {
    diag_cfg cfg = {
        .system_prompt = "You are a concise assistant.",
        .ctx_size = 16384,
        .gen_tokens = 16,
        .prefill_chunk = 0,
        .prefill_window = 0,
        .activation_bits = 0,
        .debug = true,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
    };
    parse_args(&cfg, argc, argv);

    char *prompt_text = cfg.prompt_file ? read_file(cfg.prompt_file) : strdup("Summarize this test.");
    if (!prompt_text) die("failed to read prompt");

    ds4_distributed_layers coordinator_layers = {0};
    ds4_distributed_layers worker_layers = {0};
    if (!parse_layer_spec(cfg.coordinator_layers, &coordinator_layers) ||
        !parse_layer_spec(cfg.worker_layers, &worker_layers)) {
        die("failed to parse coordinator or worker layer range");
    }

    char err[256] = {0};
    if (!set_process_lock_file("/tmp/ds4-phase4-diagnose.lock", err, sizeof(err))) {
        die(err);
    }

    ds4_engine *dist_engine = NULL;
    ds4_session *handoff_session = NULL;
    ds4_session *reference_session = NULL;
    ds4_tokens prompt = {0};
    token_trace_run handoff_run = {0};
    token_trace_run reference_run = {0};
    float *seed_logits_handoff = NULL;
    float *seed_logits_ref = NULL;
    int local_first_token = -1;
    int ref_first_token = -1;
    int mismatch_step = -1;
    bool token_match = false;
    double prefill_handoff_sec = 0.0;
    double prefill_ref_sec = 0.0;
    double shard_load_sec = 0.0;
    double local_decode_sec = 0.0;
    uint32_t route_hops = 0;
    bool output_on_coordinator = false;
    char route_summary[1024] = {0};

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
        die(err[0] ? err : "distributed options failed");
    }
    if (ds4_engine_open(&dist_engine, &dist_opt) != 0 || !dist_engine) {
        die("failed to open distributed coordinator engine");
    }

    const int vocab = ds4_engine_vocab_size(dist_engine);
    if (vocab <= 0) die("invalid vocab size");

    if (!open_distributed_session(&cfg,
                                  dist_engine,
                                  &handoff_session,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        die("failed to open handoff session");
    }
    if (output_on_coordinator) die("phase4 diagnose requires worker-owned output head");

    ds4_encode_chat_prompt(dist_engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) die("prompt tokenization failed");
    if (prompt.len >= cfg.ctx_size) die("prompt exceeds ctx");

    double t0 = now_sec();
    if (ds4_session_sync(handoff_session, &prompt, err, sizeof(err)) != 0) {
        die(err[0] ? err : "distributed prefill failed");
    }
    prefill_handoff_sec = now_sec() - t0;

    seed_logits_handoff = malloc((size_t)vocab * sizeof(seed_logits_handoff[0]));
    handoff_run.tokens = malloc((size_t)cfg.gen_tokens * sizeof(handoff_run.tokens[0]));
    handoff_run.trace = malloc((size_t)cfg.gen_tokens * (size_t)vocab * sizeof(handoff_run.trace[0]));
    if (!seed_logits_handoff || !handoff_run.tokens || !handoff_run.trace) {
        die("out of memory allocating handoff trace buffers");
    }
    if (ds4_session_copy_logits(handoff_session, seed_logits_handoff, vocab) != vocab) {
        die("failed to copy handoff seed logits");
    }
    local_first_token = ds4_session_argmax(handoff_session);
    handoff_run.token_count = ds4_session_distributed_handoff_argmax_trace(
        handoff_session,
        cfg.gen_tokens,
        handoff_run.tokens,
        cfg.gen_tokens,
        handoff_run.trace,
        cfg.gen_tokens * vocab,
        &handoff_run.trace_steps,
        &shard_load_sec,
        &local_decode_sec,
        err,
        sizeof(err));
    handoff_run.elapsed_sec = local_decode_sec;
    if (handoff_run.token_count < 0) {
        die(err[0] ? err : "distributed handoff trace failed");
    }

    ds4_session_free(handoff_session);
    handoff_session = NULL;

    if (!open_distributed_session(&cfg,
                                  dist_engine,
                                  &reference_session,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        die("failed to open reference session");
    }
    t0 = now_sec();
    if (ds4_session_sync(reference_session, &prompt, err, sizeof(err)) != 0) {
        die(err[0] ? err : "distributed reference prefill failed");
    }
    prefill_ref_sec = now_sec() - t0;

    seed_logits_ref = malloc((size_t)vocab * sizeof(seed_logits_ref[0]));
    if (!seed_logits_ref) die("out of memory allocating reference seed logits");
    if (ds4_session_copy_logits(reference_session, seed_logits_ref, vocab) != vocab) {
        die("failed to copy reference seed logits");
    }
    ref_first_token = ds4_session_argmax(reference_session);
    if (!capture_reference_trace(reference_session, vocab, cfg.gen_tokens, &reference_run)) {
        die("failed to capture distributed reference trace");
    }

    token_match = handoff_run.token_count == reference_run.token_count;
    if (token_match) {
        for (int i = 0; i < handoff_run.token_count; i++) {
            if (handoff_run.tokens[i] != reference_run.tokens[i]) {
                token_match = false;
                mismatch_step = i;
                break;
            }
        }
    }

    const float *ref_cmp = NULL;
    const float *cand_cmp = NULL;
    int local_mismatch_token = -1;
    int ref_mismatch_token = -1;
    char *local_piece = NULL;
    char *ref_piece = NULL;
    if (!token_match) {
        if (mismatch_step == 0) {
            ref_cmp = seed_logits_ref;
            cand_cmp = seed_logits_handoff;
        } else if (mismatch_step > 0 &&
                   mismatch_step - 1 < reference_run.trace_steps &&
                   mismatch_step - 1 < handoff_run.trace_steps) {
            ref_cmp = reference_run.trace + (size_t)(mismatch_step - 1) * (size_t)vocab;
            cand_cmp = handoff_run.trace + (size_t)(mismatch_step - 1) * (size_t)vocab;
        }
        if (mismatch_step >= 0 && mismatch_step < handoff_run.token_count) {
            local_mismatch_token = handoff_run.tokens[mismatch_step];
            local_piece = token_piece_text(dist_engine, local_mismatch_token);
        }
        if (mismatch_step >= 0 && mismatch_step < reference_run.token_count) {
            ref_mismatch_token = reference_run.tokens[mismatch_step];
            ref_piece = token_piece_text(dist_engine, ref_mismatch_token);
        }
    }

    const char *local_text = "";
    const char *reference_text = "";
    char *local_text_owned = tokens_render_text(dist_engine, handoff_run.tokens, handoff_run.token_count);
    char *reference_text_owned = tokens_render_text(dist_engine, reference_run.tokens, reference_run.token_count);
    if (local_text_owned) local_text = local_text_owned;
    if (reference_text_owned) reference_text = reference_text_owned;

    printf("{\n");
    printf("  \"tool\":\"issue304_phase4_diagnose\",\n");
    printf("  \"model_path\":");
    print_json_string(cfg.model_path);
    printf(",\n");
    printf("  \"prompt_file\":");
    print_json_string(cfg.prompt_file ? cfg.prompt_file : "");
    printf(",\n");
    printf("  \"ctx\":%d,\n", cfg.ctx_size);
    printf("  \"prompt_tokens\":%d,\n", prompt.len);
    printf("  \"gen_tokens\":%d,\n", cfg.gen_tokens);
    printf("  \"prefill_chunk\":%u,\n", cfg.prefill_chunk);
    printf("  \"prefill_window\":%u,\n", cfg.prefill_window);
    printf("  \"activation_bits\":%u,\n", cfg.activation_bits);
    printf("  \"route_hops\":%u,\n", route_hops);
    printf("  \"route_summary\":");
    print_json_string(route_summary);
    printf(",\n");
    printf("  \"prefill_handoff_sec\":%.6f,\n", prefill_handoff_sec);
    printf("  \"prefill_ref_sec\":%.6f,\n", prefill_ref_sec);
    printf("  \"shard_load_sec\":%.6f,\n", shard_load_sec);
    printf("  \"local_decode_sec\":%.6f,\n", local_decode_sec);
    printf("  \"reference_decode_sec\":%.6f,\n", reference_run.elapsed_sec);
    printf("  \"handoff_first_token\":%d,\n", local_first_token);
    printf("  \"reference_first_token\":%d,\n", ref_first_token);
    printf("  \"generated\":{\n");
    printf("    \"match\":%s,\n", token_match ? "true" : "false");
    printf("    \"mismatch_step\":%d,\n", mismatch_step);
    printf("    \"handoff_tokens\":");
    print_token_array(handoff_run.tokens, handoff_run.token_count);
    printf(",\n");
    printf("    \"reference_tokens\":");
    print_token_array(reference_run.tokens, reference_run.token_count);
    printf(",\n");
    printf("    \"handoff_text\":");
    print_json_string(local_text);
    printf(",\n");
    printf("    \"reference_text\":");
    print_json_string(reference_text);
    printf("\n");
    printf("  }");

    if (!token_match && ref_cmp && cand_cmp) {
        parity_metrics cmp = compare_logits(ref_cmp, cand_cmp, vocab);
        int ref_top10[10];
        int cand_top10[10];
        logits_topk(ref_cmp, vocab, ref_top10, 10);
        logits_topk(cand_cmp, vocab, cand_top10, 10);
        printf(",\n  \"mismatch\":{\n");
        printf("    \"step\":%d,\n", mismatch_step);
        printf("    \"handoff_token\":%d,\n", local_mismatch_token);
        printf("    \"reference_token\":%d,\n", ref_mismatch_token);
        printf("    \"handoff_token_text\":");
        print_json_string(local_piece ? local_piece : "");
        printf(",\n");
        printf("    \"reference_token_text\":");
        print_json_string(ref_piece ? ref_piece : "");
        printf(",\n");
        printf("    \"handoff_in_reference_top5\":%s,\n",
               topk_contains(ref_top10, 5, local_mismatch_token) ? "true" : "false");
        printf("    \"handoff_in_reference_top10\":%s,\n",
               topk_contains(ref_top10, 10, local_mismatch_token) ? "true" : "false");
        printf("    \"reference_in_handoff_top5\":%s,\n",
               topk_contains(cand_top10, 5, ref_mismatch_token) ? "true" : "false");
        printf("    \"reference_in_handoff_top10\":%s,\n",
               topk_contains(cand_top10, 10, ref_mismatch_token) ? "true" : "false");
        printf("    \"logits\":{\n");
        printf("      \"top1_ref\":%d,\n", cmp.top1_ref);
        printf("      \"top1_handoff\":%d,\n", cmp.top1_cand);
        printf("      \"top5_overlap\":%d,\n", cmp.top5_overlap);
        printf("      \"top10_overlap\":%d,\n", cmp.top10_overlap);
        printf("      \"top20_overlap\":%d,\n", cmp.top20_overlap);
        printf("      \"nonfinite\":%d,\n", cmp.nonfinite);
        printf("      \"rms\":%.8f,\n", cmp.rms);
        printf("      \"max_abs\":%.8f,\n", cmp.max_abs);
        printf("      \"top20_max_abs\":%.8f,\n", cmp.top20_max_abs);
        printf("      \"reference_top10\":");
        print_topk_json(dist_engine, ref_cmp, vocab, 10);
        printf(",\n");
        printf("      \"handoff_top10\":");
        print_topk_json(dist_engine, cand_cmp, vocab, 10);
        printf("\n");
        printf("    }\n");
        printf("  }\n");
    } else {
        printf("\n");
    }
    printf("}\n");

    free(local_piece);
    free(ref_piece);
    free(local_text_owned);
    free(reference_text_owned);
    free(seed_logits_handoff);
    free(seed_logits_ref);
    free(prompt_text);
    ds4_tokens_free(&prompt);
    free_token_trace_run(&handoff_run);
    free_token_trace_run(&reference_run);
    ds4_session_free(handoff_session);
    ds4_session_free(reference_session);
    ds4_engine_close(dist_engine);
    return 0;
}
