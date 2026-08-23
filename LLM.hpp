#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

struct CurlGlobal {
    CurlGlobal() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    ~CurlGlobal() {
        curl_global_cleanup();
    }
} curl_global;

class LLM {
    static size_t cb(void* p, size_t s, size_t n, void* u) {
        ((string*)u)->append((char*)p, s * n);
        return s * n;
    }
  public:
    string base_url;
    string api_key;
    string model;
    string system_prompt;
    long timeout_seconds;
    json extra_params;

    LLM(
        string base_url_,
        string api_key_,
        string model_,
        string system_prompt_ = "请使用平文本输出，禁止使用Markdown、Latex。",
        long timeout_seconds_ = 1800
    )
        : base_url(std::move(base_url_)),
          api_key(std::move(api_key_)),
          model(std::move(model_)),
          system_prompt(std::move(system_prompt_)),
          timeout_seconds(timeout_seconds_),
          extra_params(json::object()) {}

    virtual ~LLM() = default;

    string operator()(const string& user_prompt, double temperature = 1.0) const {
        string url = base_url + "/v1/chat/completions";

        json body = {
            {"model", model},
            {"stream", false},
            {
                "messages", {
                    {{"role", "system"}, {"content", system_prompt}},
                    {{"role", "user"}, {"content", user_prompt}}
                }
            }
        };

        if (!extra_params.empty()) {
            body.update(extra_params);
        }
        // 温度参数优先于 extra_params 中的同名字段
        body["temperature"] = temperature;

        string data = body.dump(-1, ' ', false, json::error_handler_t::replace);

        int retry_count = 0;

        while (true) {
            CURL* c = curl_easy_init();
            if (!c) {
                retry_count++;
//                cerr << "[LLM Warning] CURL init failed. Retrying... (Attempt " << retry_count << ")" << endl;
                this_thread::sleep_for(chrono::seconds(3));
                continue;
            }

            string buf;
            curl_slist* h = nullptr;
            h = curl_slist_append(h, "Content-Type: application/json");

            string auth;
            if (!api_key.empty()) {
                auth = "Authorization: Bearer " + api_key;
                h = curl_slist_append(h, auth.c_str());
            }

            curl_easy_setopt(c, CURLOPT_URL, url.c_str());
            curl_easy_setopt(c, CURLOPT_POST, 1L);
            curl_easy_setopt(c, CURLOPT_POSTFIELDS, data.c_str());
            curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, (long)data.size());
            curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, cb);
            curl_easy_setopt(c, CURLOPT_WRITEDATA, &buf);
            curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_seconds);

            long code = 0;
            CURLcode res = curl_easy_perform(c);
            curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);

            curl_slist_free_all(h);
            curl_easy_cleanup(c);

            if (res != CURLE_OK || code < 200 || code >= 300) {
                retry_count++;
//                cerr << "[LLM Warning] Network or HTTP error (Code: " << code << "). Retrying... (Attempt " << retry_count << ")" << endl;
                this_thread::sleep_for(chrono::seconds(3));
                continue;
            }

            json r = json::parse(buf, nullptr, false);
            if (r.is_discarded() || !r.contains("choices") || !r["choices"].is_array() || r["choices"].empty()) {
                retry_count++;
//                cerr << "[LLM Warning] Invalid JSON or empty response. Retrying... (Attempt " << retry_count << ")" << endl;
                this_thread::sleep_for(chrono::seconds(3));
                continue;
            }

            const json& ch = r["choices"][0];
            if (!ch.contains("message") || !ch["message"].is_object()) {
                retry_count++;
//                cerr << "[LLM Warning] Missing 'message' object. Retrying... (Attempt " << retry_count << ")" << endl;
                this_thread::sleep_for(chrono::seconds(3));
                continue;
            }

            const json& msg = ch["message"];
            if (!msg.contains("content") || !msg["content"].is_string()) {
                retry_count++;
//                cerr << "[LLM Warning] Missing 'content' string. Retrying... (Attempt " << retry_count << ")" << endl;
                this_thread::sleep_for(chrono::seconds(3));
                continue;
            }

            // --- 修改开始：处理可能存在的 </think> 标签，并去除前导空白 ---
            string content = msg["content"].get<string>();
            // 查找最后一个 </think> 标签的位置
            size_t pos = content.rfind("</think>");
            if (pos != string::npos) {
                // 截取标签之后的部分（</think> 长度 8）
                content = content.substr(pos + 8);
                // 去除前导空白字符（空格、换行、制表符等）
                size_t start = 0;
                while (start < content.size() &&
                        std::isspace(static_cast<unsigned char>(content[start]))) {
                    ++start;
                }
                content = content.substr(start);
            }
            return content;
            // --- 修改结束 ---
        }
    }
};
