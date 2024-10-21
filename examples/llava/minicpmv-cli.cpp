#include "ggml.h"
#include "log.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// extern "C" {
//     #include <libavcodec/avcodec.h>
//     #include <libavformat/avformat.h>
//     #include <libavutil/imgutils.h>
//     #include <libswscale/swscale.h>
// }

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

struct clip_image_u8 {
    int nx;
    int ny;
    std::vector<uint8_t> buf;
};

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG_TEE("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> [--video <path/to/an/video.mp4>] [--image <path/to/an/image.jpg>] [--image <path/to/another/image.jpg>] [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

static struct llama_model * llava_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(gpt_params * params) {
    auto model = llava_init(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init minicpmv model\n", __func__);
        return NULL;
    }

    const char * clip_path = params->mmproj.c_str();
    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    if (params->n_ctx < 2048) {
        // warn user here, "Image processing requires at least 2048 context, setting context to 2048"
        LOG_TEE("%s: warn: Image processing requires at least 2048 context, setting context to 2048\n" , __func__);
        ctx_params.n_ctx = 2048;
    } else {
        ctx_params.n_ctx = params->n_ctx;
    }
    
    llama_model * model2 = nullptr;
    if(params->skip_model.size() > 0 && params->skip_layers > 0) {
        //load last model
        llama_model_params model_params = llama_model_params_from_gpt_params(*params);
        model_params.init_time = false;
        model_params.has_vocab = false;
        //llama_model * model2 = llama_load_model_from_file(params->model.c_str(), model_params);
        //llama_model * model2 = llama_load_model_from_file("/Users/zkh/Downloads/last_16/ggml-model-Q4_0.gguf", model_params);
        model2 = llama_load_model_from_file(params->skip_model.c_str(), model_params);
        llama_set_model_skip_layers(model2, params->skip_layers);
        //llama_add_model_load_times(model, model2);
    }

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);
    
    if (ctx_llama == NULL) {
        LOG_TEE("%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    if(params->skip_model.size() > 0 && params->skip_layers > 0) {
        llama_set_model2(ctx_llama, model2);
    }

    for (unsigned int i = 0; i < params->lora_adapter.size(); ++i) {
        const std::string & lora_adapter = std::get<0>(params->lora_adapter[i]);
        float lora_scale = std::get<1>(params->lora_adapter[i]);
        int err = llama_model_apply_lora_from_file(model,
                                             lora_adapter.c_str(),
                                             lora_scale,
                                             ((i > 0) || params->lora_base.empty())
                                                ? NULL
                                                : params->lora_base.c_str(),
                                             params->n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return NULL;
        }
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            LOG_TEE("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

static void process_eval_image_embed(struct llava_context * ctx_llava, const struct llava_image_embed * embeds, int n_batch, int * n_past, int idx) {
    float * image_embed = (float *)malloc(clip_embd_nbytes(ctx_llava->ctx_clip));
    std::memcpy(image_embed, embeds->embed + idx * clip_n_patches(ctx_llava->ctx_clip) * clip_n_mmproj_embd(ctx_llava->ctx_clip), clip_embd_nbytes(ctx_llava->ctx_clip));

    auto slice_embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    slice_embed->embed = image_embed;
    slice_embed->n_image_pos = clip_n_patches(ctx_llava->ctx_clip);
    llava_eval_image_embed(ctx_llava->ctx_llama, slice_embed, n_batch, n_past);
    llava_image_embed_free(slice_embed);
}

static void process_eval_image_embed_l(struct llava_context * ctx_llava, const struct llava_image_embed * embeds, int n_batch, int * n_past, int idx, int sp , unsigned char * buffer) {
    int n_patches = clip_n_patches(ctx_llava->ctx_clip);
    int token_len = clip_n_mmproj_embd(ctx_llava->ctx_clip)*sizeof(float);
    float * image_embed = (float *)malloc(clip_embd_nbytes(ctx_llava->ctx_clip)+2*token_len);
    // printf("sp = %d, n_patches=%d, token_len=%d\n", sp, n_patches, token_len);
    if (sp == 0) {
        std::memcpy(image_embed, buffer, token_len);
        std::memcpy((char*)image_embed + (n_patches + 1) * token_len, buffer+token_len, token_len);
    }
    else if (sp == 1) {
        std::memcpy(image_embed, buffer+token_len*2, token_len);
        std::memcpy((char*)image_embed + (n_patches + 1) * token_len, buffer+token_len*3, token_len);
    }
    
    // printf("before memcpy, idx=%d, nbytes=%d, token_len=%d, \n", idx, clip_embd_nbytes(ctx_llava->ctx_clip), token_len);
    // printf("addr: %x\n", embeds->embed);
    std::memcpy((char*)image_embed+token_len, embeds->embed + idx * clip_n_patches(ctx_llava->ctx_clip) * clip_n_mmproj_embd(ctx_llava->ctx_clip), clip_embd_nbytes(ctx_llava->ctx_clip));

    // printf("after memcpy\n");
    auto slice_embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    slice_embed->embed = image_embed;
    slice_embed->n_image_pos = clip_n_patches(ctx_llava->ctx_clip) + 2;
    llava_eval_image_embed(ctx_llava->ctx_llama, slice_embed, n_batch, n_past);
    llava_image_embed_free(slice_embed);
}

static int process_image(struct llava_context * ctx_llava, struct llava_image_embed * embeds, gpt_params * params, int &n_past) {
    std::string system_prompt;
    bool res = false;
    int idx = 0;
    int num_image_embeds = embeds->n_image_pos / clip_n_patches(ctx_llava->ctx_clip);    
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, false);
    process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
    res = eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
    if (num_image_embeds > 1) {
        size_t num_image_embeds_col = clip_uhd_num_image_embeds_col(ctx_llava->ctx_clip);
        for (size_t i = 0; i < (num_image_embeds-1)/num_image_embeds_col; ++i) {
            for (size_t j = 0; j < num_image_embeds_col; ++j) {
                eval_string(ctx_llava->ctx_llama, std::string("<slice>").c_str(), params->n_batch, &n_past, false);
                process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
                eval_string(ctx_llava->ctx_llama, std::string("</slice>").c_str(), params->n_batch, &n_past, false);
                if (j == num_image_embeds_col - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
                }
            }
        }
    }
    res = eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    if(!res) return 0;
    return n_past;
}

static int process_image_l(struct llava_context * ctx_llava, struct llava_image_embed * embeds, gpt_params * params, int &n_past) {
    std::string system_prompt;
    bool res = false;
    int idx = 0;
    int num_image_embeds = embeds->n_image_pos / clip_n_patches(ctx_llava->ctx_clip);    
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);

    std::string fname = "D:\\project\\MinicpmModel2720\\sp.raw";
    auto file = fopen(fname.c_str(), "rb");
    if (file == NULL) {
        LOG_TEE("%s: can't read file %s\n", __func__, fname.c_str());
        return 0;
    }
    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize);
    if (buffer == NULL) {
        LOG_TEE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, fname.c_str());
        perror("Memory allocation error");
        fclose(file);
        free(buffer);
        return 0;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file);
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file);

    printf("after load sp.raw\n");
    process_eval_image_embed_l(ctx_llava, embeds, params->n_batch, &n_past, idx++, 0, buffer);
    printf("after process_eval_image_embed_l\n");
    if (num_image_embeds > 1) {
        size_t num_image_embeds_col = clip_uhd_num_image_embeds_col(ctx_llava->ctx_clip);
        for (size_t i = 0; i < (num_image_embeds-1)/num_image_embeds_col; ++i) {
            for (size_t j = 0; j < num_image_embeds_col; ++j) {
                process_eval_image_embed_l(ctx_llava, embeds, params->n_batch, &n_past, idx++, 1, buffer);
                if (j == num_image_embeds_col - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
                }
            }
        }
    }
    res = eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    free(buffer);
    if(!res) return 0;
    return n_past;
}

static bool process_prompt(int type, struct llava_context * ctx_llava, gpt_params * params, int &n_past, std::string prompt = ""){
    int has_minicpmv_projector = clip_is_minicpmv(ctx_llava->ctx_clip);
    if (type==0) {
        std::string system_prompt;
        if (has_minicpmv_projector == 1) {
            system_prompt = "<用户>";
        }
        else if (has_minicpmv_projector == 2) {
            system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
        }
        else if (has_minicpmv_projector == 3) {
            system_prompt = "<|im_start|>user\n";
        }
        else if (has_minicpmv_projector == 4) {
            system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
        }
        return eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, false);
    }
    else if (type==1) {
        std::string user_prompt = prompt;
        return eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
    }
    else if (type==2) {
        // printf("has_minicpmv_projector=%d\n", has_minicpmv_projector);
        if (has_minicpmv_projector == 1) {
            return eval_string(ctx_llava->ctx_llama, "<AI>\n", params->n_batch, &n_past, false);
        }
        else if (has_minicpmv_projector == 2) {
            return eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
        }
        else if (has_minicpmv_projector == 3) {
            return eval_string(ctx_llava->ctx_llama, "<|im_end|><|im_start|>assistant\n", params->n_batch, &n_past, false);
        }
        else if (has_minicpmv_projector == 4) {
            return eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
        }
    }
    return 0;
}

static std::vector<int> g_ids;

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    g_ids.push_back(id);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static struct llava_context * minicpmv_init(gpt_params * params, const std::string & fname, int &n_past){
    auto ctx_llava = llava_init_context(params);
    auto embeds = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, fname.c_str());
    if (!embeds) {
        LOG_TEE("error: failed to load image %s. Terminating\n\n", fname.c_str());
        return NULL;
    }

    // process the prompt
    if (params->prompt.empty() && params->interactive == false) {
        LOG_TEE("prompt should be given or interactive mode should be on");
        return NULL;
    }

    const int64_t t_llava_init_start_us = ggml_time_us();

    const int64_t t_llava_init_end_us = ggml_time_us();
    float t_llava_init_ms = (t_llava_init_end_us - t_llava_init_start_us) / 1000.0;
    LOG_TEE("\n%s: llava init in %8.2f ms.\n", __func__, t_llava_init_ms);

    const int64_t t_process_image_start_us = ggml_time_us();
    process_prompt(0, ctx_llava, params, n_past);
    process_image_l(ctx_llava, embeds, params, n_past);

    const int64_t t_process_image_end_us = ggml_time_us();
    float t_process_image_ms = (t_process_image_end_us - t_process_image_start_us) / 1000.0;
    LOG_TEE("\n%s: llama process image in %8.2f ms.\n", __func__, t_process_image_ms);

    llava_image_embed_free(embeds);
    return ctx_llava;
}

static int process_input(struct llava_context * ctx_llava, gpt_params * params, int type, std::string prompt, int &n_past, struct llava_image_embed * embeds = nullptr){
    if (type==0) {
        if (process_prompt(1, ctx_llava, params, n_past, prompt)) return 1;
    }
    else if (type == 1) {
        if(embeds != NULL){
            return (process_image(ctx_llava, embeds, params, n_past));
        }
    }
    return 0;
}

static struct llama_sampling_context * llama_init(struct llava_context * ctx_llava, gpt_params * params, std::string prompt, int &n_past, bool is_first = false){
    std::string user_prompt = prompt;
    if(is_first)process_prompt(0, ctx_llava, params, n_past);
    process_prompt(1, ctx_llava, params, n_past, prompt);
    process_prompt(2, ctx_llava, params, n_past);

    // generate the response
    LOG_TEE("\n");
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    return ctx_sampling;
}

static const char * llama_loop(struct llava_context * ctx_llava,struct llama_sampling_context * ctx_sampling, int &n_past){

    const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
    return tmp;
}

#include <assert.h>

// Used for debugging to print out beam tokens.
struct ostream_beam_view {
    llama_context * ctx;
    llama_beam_view beam_view;
};

static std::ostream & operator<<(std::ostream & os, const ostream_beam_view & obv) {
    os << "p(" << obv.beam_view.p << ") eob(" << std::boolalpha << obv.beam_view.eob << ") tokens(";
    for (size_t i = 0 ; i < obv.beam_view.n_tokens ; ++i) {
        os << llama_token_to_piece(obv.ctx, obv.beam_view.tokens[i]);
    }
    return os << ')';
}

struct beam_search_callback_data {
    llama_context * ctx;
    std::vector<llama_token> response;
    std::vector<llama_token> response1;
    std::vector<llama_token> response2;
    std::vector<llama_token> response3;
};

// In this case, end-of-beam (eob) is equivalent to end-of-sentence (eos) but this need not always be the same.
// For example, eob can be flagged due to maximum token length, stop words, etc.
static bool is_at_eob(const beam_search_callback_data & callback_data, const llama_token * tokens, size_t n_tokens) {
    return n_tokens && llama_token_is_eog(llama_get_model(callback_data.ctx), tokens[n_tokens-1]);
}

// Function matching type llama_beam_search_callback_fn_t.
// Custom callback example is called each time the beams lengths increase:
//  * Show progress by printing ',' following by number of convergent beam tokens if any.
//  * When all beams converge to a common prefix, they are made available in beams_state.beams[0].
//    This is also called when the stop condition is met.
//    Collect tokens into std::vector<llama_token> response which is pointed to by callback_data.
static void beam_search_callback(void * callback_data_ptr, llama_beams_state beams_state) {
    auto& callback_data = *static_cast<beam_search_callback_data*>(callback_data_ptr);
    // Mark beams as EOS as needed.
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        llama_beam_view& beam_view = beams_state.beam_views[i];
        if (!beam_view.eob && is_at_eob(callback_data, beam_view.tokens, beam_view.n_tokens)) {
            beam_view.eob = true;
        }
    }
    //printf(",");  // Show progress
    if (const size_t n = beams_state.common_prefix_length) {
        // callback_data.response.resize(callback_data.response.size() + n);
        // callback_data.response1.resize(callback_data.response1.size() + n);
        // callback_data.response2.resize(callback_data.response2.size() + n);
        // callback_data.response3.resize(callback_data.response3.size() + n);
        assert(0u < beams_state.n_beams);
        const llama_token * tokens = beams_state.beam_views[0].tokens;
        // std::copy(tokens, tokens + n, callback_data.response.end() - n);
        // tokens = beams_state.beam_views[1].tokens;
        // std::copy(tokens, tokens + n, callback_data.response1.end() - n);
        // tokens = beams_state.beam_views[2].tokens;
        // std::copy(tokens, tokens + n, callback_data.response2.end() - n);
        // tokens = beams_state.beam_views[3].tokens;
        // std::copy(tokens, tokens + n, callback_data.response3.end() - n);
        //printf("%zu", n);
        for(int i = 0; i < n; i++){
            std::cout << llama_token_to_piece(callback_data.ctx, tokens[i]) << std::flush;
        }
    }
    //fflush(stdout);
#if 1 // DEBUG: print current beams for this iteration
    // std::cout << "\n\nCurrent beams (last_call=" << beams_state.last_call << "):\n";
    // for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
    //     std::cout << "beams["<<i<<"]: " << ostream_beam_view{callback_data.ctx,beams_state.beam_views[i]} << std::endl;
    // }
   
#endif
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }

    params.n_beams = 1;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("llava", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    if (params.mmproj.empty()) {
        gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }
    
    int n_past = 0;
    struct llava_context * ctx_llava = nullptr;

        if (params.image.size() > 0) {
            auto image = params.image;
            ctx_llava = minicpmv_init(&params, image, n_past);
            //release vit memory
            //clip_free(ctx_llava->ctx_clip);
            if (!params.prompt.empty()) {
                LOG_TEE("<user>%s\n", params.prompt.c_str());
                LOG_TEE("<assistant>");
                auto ctx_sampling = llama_init(ctx_llava, &params, params.prompt.c_str(), n_past, false);
                if(params.n_beams == 0){
                    const int max_tgt_len = params.n_predict < 0 ? 8192 : params.n_predict;
                    std::string response = "";
                    bool have_tmp = false;
                    for (int i = 0; i < max_tgt_len; i++) {
                        auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                        response += tmp;
                        if (strcmp(tmp, "</s>") == 0){
                            if(!have_tmp)continue;
                            else break;
                        }
                        have_tmp = true;
                        printf("%s", tmp);
                        if (strstr(response.c_str(), "<user>")) break; // minicpm-v 

                        fflush(stdout);
                    }
                    llama_sampling_free(ctx_sampling);
                }else{
                    beam_search_callback_data callback_data{ctx_llava->ctx_llama, {}};
                    size_t const beam_width = static_cast<size_t>(params.n_beams);
                    int const n_predict = params.n_predict;
                    llama_beam_search(ctx_llava->ctx_llama, beam_search_callback, &callback_data, beam_width, n_past, n_predict);

                    // std::cout << "\n\n";
                    // for (llama_token const token_id : callback_data.response) {
                    //     std::cout << llama_token_to_piece(ctx_llava->ctx_llama, token_id);
                    // }
                    // std::cout << std::endl;
                }
               
                

                // std::cout << "\n\n";
                // for (llama_token const token_id : callback_data.response1) {
                //     std::cout << llama_token_to_piece(ctx_llava->ctx_llama, token_id);
                // }
                // std::cout << std::endl;

                // std::cout << "\n\n";
                // for (llama_token const token_id : callback_data.response2) {
                //     std::cout << llama_token_to_piece(ctx_llava->ctx_llama, token_id);
                // }
                // std::cout << std::endl;

                // std::cout << "\n\n";
                // for (llama_token const token_id : callback_data.response3) {
                //     std::cout << llama_token_to_piece(ctx_llava->ctx_llama, token_id);
                // }
                // std::cout << std::endl;
            }
            else {
                while (true) {
                    LOG_TEE("<user>");
                    std::string prompt;
                    std::getline(std::cin, prompt);
                    LOG_TEE("<assistant>");
                    auto ctx_sampling = llama_init(ctx_llava, &params, prompt, n_past, true);
                    const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
                    std::string response = "";
                    for (int i = 0; i < max_tgt_len; i++) {
                        auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                        response += tmp;
                        if (strcmp(tmp, "</s>") == 0) break;
                        if (strstr(tmp, "###")) break; // Yi-VL behavior
                        printf("%s", tmp);// mistral llava-1.6
                        if (strstr(response.c_str(), "<user>")) break; // minicpm-v 
                        fflush(stdout);
                    }
                    llama_sampling_free(ctx_sampling);
                }                
            }
        }
        printf("\n");
        llama_print_timings(ctx_llava->ctx_llama);

        for(int i = 0; i < g_ids.size(); i++){
            std::cout << g_ids[i] << " ";
        }
        std::cout << std::endl;

        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    // }

    return 0;
}
