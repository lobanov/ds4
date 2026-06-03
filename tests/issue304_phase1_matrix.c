#include "../ds4.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *mode;
    const char *model_path;
    const char *prompt_file;
    const char *payload_path;
    const char *logits_path;
    const char *tokens_path;
    const char *trace_path;
    const char *ref_trace_path;
    int ctx;
    int frontier;
    int greedy_steps;
} matrix_cfg;

typedef struct {
    int top1_saved;
    int top1_restored;
    int overlap;
    int top5_overlap;
    int nonfinite;
    float rms;
    float max_abs;
} logits_cmp;

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase1-matrix: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "issue304-phase1-matrix: %s: %s\n", msg, strerror(errno));
    exit(1);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --mode save|load-check --model FILE --payload FILE --logits FILE --tokens FILE "
            "[--prompt-file FILE] [--ctx N] [--frontier N] [--greedy-steps N]\n",
            argv0);
    exit(2);
}

static char *read_file(const char *path, size_t *len_out) {
    if (len_out) *len_out = 0;
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
    if (len_out) *len_out = (size_t)len;
    return buf;
}

static bool write_binary_file(const char *path, const void *ptr, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool ok = fwrite(ptr, 1, bytes, fp) == bytes;
    ok = ok && fclose(fp) == 0;
    return ok;
}

static bool read_binary_file(const char *path, void *ptr, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    bool ok = fread(ptr, 1, bytes, fp) == bytes;
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

static bool read_tokens_file(const char *path, int *tokens, int cap, int *count_out) {
    *count_out = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    int count = 0;
    if (fscanf(fp, "%d", &count) != 1 || count < 0 || count > cap) {
        fclose(fp);
        return false;
    }
    for (int i = 0; i < count; i++) {
        if (fscanf(fp, "%d", &tokens[i]) != 1) {
            fclose(fp);
            return false;
        }
    }
    fclose(fp);
    *count_out = count;
    return true;
}

static bool read_trace_file(const char *path, float *ptr, size_t bytes) {
    return read_binary_file(path, ptr, bytes);
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

static bool capture_greedy_tokens(ds4_session *session, int steps, int *out, int *out_len) {
    char err[160];
    int n = 0;
    while (n < steps) {
        int token = ds4_session_argmax(session);
        out[n++] = token;
        if (n < steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            *out_len = n;
            return false;
        }
    }
    *out_len = n;
    return true;
}

static void parse_args(matrix_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            cfg->mode = argv[++i];
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--payload") && i + 1 < argc) {
            cfg->payload_path = argv[++i];
        } else if (!strcmp(argv[i], "--logits") && i + 1 < argc) {
            cfg->logits_path = argv[++i];
        } else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            cfg->tokens_path = argv[++i];
        } else if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            cfg->trace_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-trace") && i + 1 < argc) {
            cfg->ref_trace_path = argv[++i];
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            cfg->ctx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--frontier") && i + 1 < argc) {
            cfg->frontier = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--greedy-steps") && i + 1 < argc) {
            cfg->greedy_steps = atoi(argv[++i]);
        } else {
            usage(argv[0]);
        }
    }
}

static ds4_engine *open_engine(const matrix_cfg *cfg) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = cfg->model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
    };
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        die("failed to open engine");
    }
    return engine;
}

static bool tokenize_prompt(ds4_engine *engine, const char *prompt_file, ds4_tokens *prompt) {
    memset(prompt, 0, sizeof(*prompt));
    char *text = read_file(prompt_file, NULL);
    if (!text) return false;
    ds4_tokenize_text(engine, text, prompt);
    free(text);
    return prompt->len > 0;
}

static int run_save(const matrix_cfg *cfg) {
    ds4_engine *engine = open_engine(cfg);
    ds4_tokens prompt = {0};
    if (!tokenize_prompt(engine, cfg->prompt_file, &prompt)) die_errno("read prompt");
    if (cfg->frontier <= 0 || cfg->frontier > prompt.len) die("invalid frontier");

    ds4_tokens prefix = {.v = prompt.v, .len = cfg->frontier, .cap = cfg->frontier};
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->ctx) != 0 || !session) die("failed to create session");

    char err[160];
    if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase1-matrix: sync failed: %s\n", err);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) die("out of memory for logits");
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) die("failed to copy logits");

    ds4_session_payload_file payload = {0};
    if (ds4_session_stage_payload(session, &payload, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase1-matrix: stage failed: %s\n", err);
        return 1;
    }

    int *tokens = malloc((size_t)cfg->greedy_steps * sizeof(tokens[0]));
    int token_count = 0;
    if (!tokens) die("out of memory for greedy tokens");
    if (!capture_greedy_tokens(session, cfg->greedy_steps, tokens, &token_count)) {
        die("failed to capture greedy tokens");
    }

    FILE *dst = fopen(cfg->payload_path, "wb");
    if (!dst) die_errno("open payload output");
    if (ds4_session_write_staged_payload(&payload, dst, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase1-matrix: write staged payload failed: %s\n", err);
        fclose(dst);
        return 1;
    }
    if (fclose(dst) != 0) die_errno("close payload output");
    if (!write_binary_file(cfg->logits_path, logits, (size_t)vocab * sizeof(logits[0]))) {
        die_errno("write logits output");
    }
    if (!write_tokens_file(cfg->tokens_path, tokens, token_count)) {
        die_errno("write tokens output");
    }

    printf("{\"mode\":\"save\",\"backend\":\"%s\",\"ctx\":%d,\"frontier\":%d,\"payload_bytes\":%llu,\"vocab\":%d,\"greedy_steps\":%d}\n",
#ifdef __APPLE__
           "metal",
#else
           "cuda",
#endif
           cfg->ctx,
           cfg->frontier,
           (unsigned long long)payload.bytes,
           vocab,
           token_count);

    free(tokens);
    free(logits);
    ds4_session_payload_file_free(&payload);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}

static int run_load_check(const matrix_cfg *cfg) {
    ds4_engine *engine = open_engine(cfg);
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->ctx) != 0 || !session) die("failed to create session");

    FILE *fp = fopen(cfg->payload_path, "rb");
    if (!fp) die_errno("open payload input");
    if (fseek(fp, 0, SEEK_END) != 0) die_errno("seek payload input");
    long bytes = ftell(fp);
    if (bytes < 0) die_errno("measure payload input");
    rewind(fp);

    char err[160];
    if (ds4_session_load_payload(session, fp, (uint64_t)bytes, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase1-matrix: load failed: %s\n", err);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    const int vocab = ds4_engine_vocab_size(engine);
    float *saved_logits = malloc((size_t)vocab * sizeof(saved_logits[0]));
    float *restored_logits = malloc((size_t)vocab * sizeof(restored_logits[0]));
    int *saved_tokens = malloc((size_t)cfg->greedy_steps * sizeof(saved_tokens[0]));
    int *restored_tokens = malloc((size_t)cfg->greedy_steps * sizeof(restored_tokens[0]));
    int saved_token_count = 0;
    int restored_token_count = 0;
    int greedy_mismatch_step = -1;
    int greedy_saved_token = -1;
    int greedy_restored_token = -1;
    if (!saved_logits || !restored_logits || !saved_tokens || !restored_tokens) die("out of memory in load-check");

    if (!read_binary_file(cfg->logits_path, saved_logits, (size_t)vocab * sizeof(saved_logits[0]))) {
        die_errno("read logits input");
    }
    if (ds4_session_copy_logits(session, restored_logits, vocab) != vocab) die("failed to copy restored logits");
    if (!read_tokens_file(cfg->tokens_path, saved_tokens, cfg->greedy_steps, &saved_token_count)) {
        die_errno("read tokens input");
    }
    if (!capture_greedy_tokens(session, cfg->greedy_steps, restored_tokens, &restored_token_count)) {
        die("failed to capture restored greedy tokens");
    }

    logits_cmp cmp = compare_logits(saved_logits, restored_logits, vocab);
    bool tokens_match = saved_token_count == restored_token_count;
    for (int i = 0; i < saved_token_count && i < restored_token_count; i++) {
        if (saved_tokens[i] != restored_tokens[i]) {
            tokens_match = false;
            if (greedy_mismatch_step < 0) {
                greedy_mismatch_step = i;
                greedy_saved_token = saved_tokens[i];
                greedy_restored_token = restored_tokens[i];
            }
        }
    }

    printf("{\"mode\":\"load-check\",\"backend\":\"%s\",\"ctx\":%d,\"payload_bytes\":%ld,\"top1_saved\":%d,\"top1_restored\":%d,\"top5_overlap\":%d,\"top20_overlap\":%d,\"rms\":%.9g,\"max_abs\":%.9g,\"nonfinite\":%d,\"greedy_steps\":%d,\"greedy_match\":%s,\"greedy_mismatch_step\":%d,\"greedy_saved_token\":%d,\"greedy_restored_token\":%d}\n",
#ifdef __APPLE__
           "metal",
#else
           "cuda",
#endif
           cfg->ctx,
           bytes,
           cmp.top1_saved,
           cmp.top1_restored,
           cmp.top5_overlap,
           cmp.overlap,
           cmp.rms,
           cmp.max_abs,
           cmp.nonfinite,
           restored_token_count,
           tokens_match ? "true" : "false",
           greedy_mismatch_step,
           greedy_saved_token,
           greedy_restored_token);

    free(saved_tokens);
    free(restored_tokens);
    free(saved_logits);
    free(restored_logits);
    ds4_session_free(session);
    ds4_engine_close(engine);

    if (cmp.nonfinite != 0 || cmp.top1_saved != cmp.top1_restored ||
        cmp.top5_overlap < 5 || cmp.overlap < 16 || !tokens_match) {
        return 1;
    }
    return 0;
}

static int run_trace_forced(const matrix_cfg *cfg) {
    ds4_engine *engine = open_engine(cfg);
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg->ctx) != 0 || !session) die("failed to create session");

    FILE *fp = fopen(cfg->payload_path, "rb");
    if (!fp) die_errno("open payload input");
    if (fseek(fp, 0, SEEK_END) != 0) die_errno("seek payload input");
    long payload_bytes = ftell(fp);
    if (payload_bytes < 0) die_errno("measure payload input");
    rewind(fp);

    char err[160];
    if (ds4_session_load_payload(session, fp, (uint64_t)payload_bytes, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase1-matrix: load failed: %s\n", err);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    int *tokens = malloc((size_t)cfg->greedy_steps * sizeof(tokens[0]));
    if (!tokens) die("out of memory for trace tokens");
    int token_count = 0;
    if (!read_tokens_file(cfg->tokens_path, tokens, cfg->greedy_steps, &token_count)) {
        die_errno("read tokens input");
    }

    const int vocab = ds4_engine_vocab_size(engine);
    const size_t trace_floats = (size_t)(token_count + 1) * (size_t)vocab;
    float *trace = malloc(trace_floats * sizeof(trace[0]));
    if (!trace) die("out of memory for trace output");

    if (ds4_session_copy_logits(session, trace, vocab) != vocab) die("failed to copy initial trace logits");
    for (int i = 0; i < token_count; i++) {
        if (ds4_session_eval(session, tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase1-matrix: forced eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            return 1;
        }
        if (ds4_session_copy_logits(session, trace + (size_t)(i + 1) * (size_t)vocab, vocab) != vocab) {
            die("failed to copy trace logits");
        }
    }

    if (!write_binary_file(cfg->trace_path, trace, trace_floats * sizeof(trace[0]))) {
        die_errno("write trace output");
    }

    printf("{\"mode\":\"trace-forced\",\"backend\":\"%s\",\"ctx\":%d,\"payload_bytes\":%ld,\"steps\":%d,\"vocab\":%d}\n",
#ifdef __APPLE__
           "metal",
#else
           "cuda",
#endif
           cfg->ctx,
           payload_bytes,
           token_count,
           vocab);

    free(trace);
    free(tokens);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}

static int run_compare_trace(const matrix_cfg *cfg) {
    int *tokens = malloc((size_t)cfg->greedy_steps * sizeof(tokens[0]));
    if (!tokens) die("out of memory for compare tokens");
    int token_count = 0;
    if (!read_tokens_file(cfg->tokens_path, tokens, cfg->greedy_steps, &token_count)) {
        die_errno("read tokens input");
    }

    ds4_engine *engine = open_engine(cfg);
    const int vocab = ds4_engine_vocab_size(engine);
    ds4_engine_close(engine);

    const size_t trace_floats = (size_t)(token_count + 1) * (size_t)vocab;
    const size_t trace_bytes = trace_floats * sizeof(float);
    float *ref = malloc(trace_bytes);
    float *cand = malloc(trace_bytes);
    if (!ref || !cand) die("out of memory for trace compare");
    if (!read_trace_file(cfg->ref_trace_path, ref, trace_bytes)) die_errno("read ref trace");
    if (!read_trace_file(cfg->trace_path, cand, trace_bytes)) die_errno("read candidate trace");

    int first_bad_step = -1;
    logits_cmp first_cmp = {0};
    for (int step = 0; step <= token_count; step++) {
        logits_cmp cmp = compare_logits(ref + (size_t)step * (size_t)vocab,
                                        cand + (size_t)step * (size_t)vocab,
                                        vocab);
        if (cmp.nonfinite != 0 || cmp.top1_saved != cmp.top1_restored ||
            cmp.top5_overlap < 5 || cmp.overlap < 16 || cmp.rms != 0.0f || cmp.max_abs != 0.0f) {
            first_bad_step = step;
            first_cmp = cmp;
            break;
        }
    }

    printf("{\"mode\":\"compare-trace\",\"steps\":%d,\"vocab\":%d,\"first_bad_step\":%d",
           token_count, vocab, first_bad_step);
    if (first_bad_step >= 0) {
        printf(",\"top1_ref\":%d,\"top1_cand\":%d,\"top5_overlap\":%d,\"top20_overlap\":%d,\"rms\":%.9g,\"max_abs\":%.9g,\"forced_token_before_step\":%d",
               first_cmp.top1_saved,
               first_cmp.top1_restored,
               first_cmp.top5_overlap,
               first_cmp.overlap,
               first_cmp.rms,
               first_cmp.max_abs,
               first_bad_step > 0 ? tokens[first_bad_step - 1] : -1);
    }
    printf("}\n");

    free(ref);
    free(cand);
    free(tokens);
    return first_bad_step >= 0 ? 1 : 0;
}

int main(int argc, char **argv) {
    matrix_cfg cfg = {
        .mode = NULL,
        .model_path = NULL,
        .prompt_file = "tests/long_context_story_prompt.txt",
        .payload_path = NULL,
        .logits_path = NULL,
        .tokens_path = NULL,
        .trace_path = NULL,
        .ref_trace_path = NULL,
        .ctx = 192,
        .frontier = 129,
        .greedy_steps = 8,
    };
    parse_args(&cfg, argc, argv);
    if (!cfg.mode || !cfg.model_path) usage(argv[0]);
    if (!strcmp(cfg.mode, "save")) {
        if (!cfg.payload_path || !cfg.logits_path || !cfg.tokens_path) usage(argv[0]);
        return run_save(&cfg);
    }
    if (!strcmp(cfg.mode, "load-check")) {
        if (!cfg.payload_path || !cfg.logits_path || !cfg.tokens_path) usage(argv[0]);
        return run_load_check(&cfg);
    }
    if (!strcmp(cfg.mode, "trace-forced")) {
        if (!cfg.payload_path || !cfg.tokens_path || !cfg.trace_path) usage(argv[0]);
        return run_trace_forced(&cfg);
    }
    if (!strcmp(cfg.mode, "compare-trace")) {
        if (!cfg.tokens_path || !cfg.trace_path || !cfg.ref_trace_path) usage(argv[0]);
        return run_compare_trace(&cfg);
    }
    usage(argv[0]);
    return 2;
}
