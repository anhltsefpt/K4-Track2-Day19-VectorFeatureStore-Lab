# Bonus — Hybrid Memory cho trợ lý AI tiếng Việt

**Contributors:** Lê Tuấn Anh (2A202602952) — làm một mình, có dùng Claude Code
(xem *Vibe-coding log* ở cuối).
**Code:** `bonus/agent.py` (`HybridMemoryAgent`), `bonus/demo.py` (`python bonus/demo.py`, exit 0).

---

## 1. Kiến trúc tổng quan

Trợ lý cần trả lời hai câu hỏi khác nhau cho mỗi request: **"cái gì liên quan?"**
(episodic memory, nằm trong vector store) và **"người này là ai?"** (profile +
hoạt động gần đây, nằm trong feature store). Hai nguồn được ghép thành một context
string trước khi đưa vào LLM.

```
                       WRITE PATH                                   READ PATH
 ┌──────────────┐                                   ┌──────────────────────────────┐
 │ hội thoại /  │  remember(text, user_id, topic)   │ recall(query, user_id)       │
 │ tài liệu /   │──────────┐                        └──────┬───────────────┬───────┘
 │ ghi chú      │          ▼                               │               │
 └──────────────┘   chunk ~200 token                       │               │
                    (câu, overlap 1 câu)                   ▼               ▼
                           │                  ┌──────────────────┐  ┌────────────────────┐
                           ▼                  │ FEAST online     │  │ QDRANT `memories`  │
                    embed (fastembed)         │ (SQLite / Redis) │  │ filter user_id     │
                           │                  │ user_profile     │  │  ├ BM25 (vi-norm)  │
                           ▼   upsert ngay    │  TTL 30d, daily  │  │  ├ vector cosine   │
                 ┌──────────────────────┐     │ query_velocity   │  │  └ profile list    │
                 │ QDRANT `memories`    │     │  TTL 1h, push    │  │   (topic_affinity) │
                 │ payload: user_id,    │     └────────┬─────────┘  └─────────┬──────────┘
                 │ text, topic          │              │ topic_affinity ──────┤
                 └──────────────────────┘              │                      ▼
                                                       │          weighted RRF (k=60)
 batch hằng ngày ──► offline store (Parquet) ──►       │          1.0·BM25 + 1.0·vec + 0.3·profile
 materialize ──► Feast online                          ▼                      │
                                         ┌─────────────────────────────────────────┐
                                         │ context: "User likes cloud reading at   │
                                         │ 187wpm. Recent activity: 11 queries ... │
                                         │ Top memories: 1. ... 2. ... 3. ..."     │
                                         └────────────────────┬────────────────────┘
                                                              ▼
                                                        LLM final response
```

---

## 2. Ba quyết định kiến trúc

### Quyết định 1 — Chunking: theo câu, gộp ~200 token, overlap 1 câu

`chunk_text()` tách văn bản theo câu và xuống dòng, rồi gộp các câu liên tiếp
đến tối đa 150 từ. Với tiếng Việt, 150 từ ≈ 200 token vì mỗi âm tiết thường bị
tokenizer cắt thành 1–2 token. Câu cuối của chunk trước được lặp lại ở đầu chunk
sau.

| Phương án | Chất lượng tìm | Chi phí lưu | Context window |
|---|---|---|---|
| Mỗi tin nhắn 1 chunk | Nhiễu: "ok", "cảm ơn" thành vector riêng | Nhiều vector nhất | Thiếu ngữ cảnh, phải lấy thêm |
| **~200 token theo câu (chọn)** | Mỗi chunk thường chứa một ý trọn vẹn | Vừa phải | 3 chunk ≈ 600 token, vừa prompt |
| Cả hội thoại 1 chunk | Vector bị "pha loãng" nhiều chủ đề | Ít nhất | Nhồi cả hội thoại vào prompt |
| Semantic break | Tốt nhất | Phải embed từng câu để đo | Tốt, nhưng tính toán đắt |

**Vì sao chọn:** cân bằng tốt nhất giữa ba trục, không cần embed thêm như
semantic break. Overlap 1 câu để một ý nằm ở ranh giới hai chunk vẫn tìm được.
**Đánh đổi:** chunk có thể cắt giữa hai chủ đề, điều mà semantic break tránh được.

### Quyết định 2 — Feature schema: tabular, tái dùng 3 feature view của NB4

| Feature | Entity | TTL | Nguồn | Dùng để |
|---|---|---|---|---|
| `topic_affinity`, `preferred_language`, `reading_speed_wpm` | `user` | 30 ngày | batch hằng ngày | cá nhân hoá, độ dài câu trả lời |
| `queries_last_hour`, `distinct_topics_24h` | `user` | 1 giờ | streaming (push) | phát hiện mối quan tâm hiện tại |
| `click_count_24h`, `ctr_7d` | `item` | 24 giờ | batch theo giờ | (chưa dùng trong POC) |

**Vì sao tabular thay vì embedding feature:** feature dạng cột *giải thích được*.
Khi user hỏi "vì sao bạn gợi ý bài này?", trợ lý trả lời được "vì bạn quan tâm
cloud". Feature dạng cột cũng dễ debug và kiểm tra point-in-time (NB4/NB8). Một
embedding feature (vector "sở thích ẩn" tính từ lịch sử đọc) bắt được sở thích
tinh tế hơn, nhưng là hộp đen, và phải tính lại toàn bộ mỗi khi đổi embedding
model, giống bài học "đổi model = index lại" ở NB2.

**Cách dùng profile trong retrieval:** profile thành **danh sách xếp hạng thứ 3**
trong RRF (các memory có `topic == topic_affinity`), với **trọng số 0.3**.
Lần chạy đầu tôi để trọng số 1.0, và *mọi* query đều trả về cùng 3 memory về
cloud, kể cả "Tôi đã đọc gì về Kubernetes?". Ở trọng số 0.3, profile chỉ còn
vai trò phá thế hoà: query cụ thể vẫn do BM25 và vector quyết định, còn query
chung chung ("Recommend đọc gì tiếp") được đẩy về chủ đề user thích.

### Quyết định 3 — Freshness: phân tầng theo loại dữ liệu

| Use case | Độ tươi cần | Cơ chế | Vì sao |
|---|---|---|---|
| User vừa lưu ghi chú rồi hỏi lại ngay | **Tức thì** | `remember()` upsert thẳng vào Qdrant | Nếu trợ lý "quên" thứ vừa lưu, user mất niềm tin ngay |
| "Tôi đang quan tâm gì gần đây?" | **Vài giây** | Feast push API → `query_velocity_features` (TTL 1h). POC: buffer `recent` trong phiên | Tín hiệu hết giá trị sau vài giờ; TTL 1h tự loại dữ liệu cũ |
| Profile ổn định (ngôn ngữ, chủ đề, tốc độ đọc) | **Hằng ngày** | batch → offline store → `materialize` | Ít thay đổi; streaming chỉ thêm chi phí hạ tầng |

Demo kiểm chứng tầng 1: ghi memory "Karpenter" rồi `recall("Karpenter là gì?")`
ngay sau đó trả về memory đó ở vị trí số 1.

**Liên kết với lab:** TTL 1h / 24h / 30 ngày lấy đúng từ `feature_views.py`
(NB4). Khi huấn luyện mô hình gợi ý từ log, profile phải lấy bằng **PIT join**
(`get_historical_features`). Nếu lấy giá trị mới nhất, mô hình được thấy dữ liệu
"tương lai". Lỗi này tôi đã gặp thật ở NB4, khi `u_001` bị loại khỏi kết quả vì
được hỏi trước thời điểm feature tồn tại.

---

## 3. Phương án đã loại

- **Lưu episodic memory trong Feast (embedding feature view), không dùng Qdrant.**
  Loại vì hai loại dữ liệu có nhịp cập nhật khác hẳn nhau (memory mới mỗi phút,
  profile theo ngày), và Feast online store tra theo *khóa*, không tìm được theo
  *độ giống nhau*.
- **Mỗi user một collection Qdrant.** Cách ly tuyệt đối, nhưng hàng triệu user
  thì hàng triệu collection, mỗi collection một HNSW index, quá tốn tài nguyên.
  Chọn **một collection + filter `user_id` ở mọi lệnh đọc** (filtered ANN, NB5).
  Demo có memory bí mật của `u_002` và `assert` rằng nó không bao giờ xuất hiện
  trong context của `u_001`, đúng bài học rò dữ liệu giữa các tenant ở NB7.
- **pyvi / underthesea để tách từ.** Tách đúng từ ghép ("đám_mây") nhưng thêm
  dependency nặng, chậm hơn, và vẫn hỏng với câu trộn Anh–Việt.

---

## 4. Lưu ý riêng cho người dùng Việt Nam

- **Gõ không dấu:** `tokenize_vi()` index thêm bản không dấu của mỗi token
  ("tự động" → "tu", "dong"), và đổi "đ" → "d". Nhờ vậy query "tu dong mo rong"
  vẫn khớp được tài liệu có dấu. Đánh đổi: tăng va chạm nghĩa ("ma" khớp cả "mà",
  "má", "mã").
- **Code-switching:** user Việt hay viết "recommend đọc gì", "summary cloud
  security". Tách theo khoảng trắng giữ nguyên thuật ngữ tiếng Anh, nên BM25 vẫn
  bắt được "Kubernetes", "IAM", "MFA".
- **Embedding model:** `bge-small-en` hiểu kém tiếng Việt. Trong demo, query
  "tự động mở rộng hạ tầng" được vector xếp *công thức phở* lên đầu (cosine 0.70).
  BM25 cứu được kết quả. Bản thật nên dùng `bge-m3` (`EMBEDDING_BACKEND=bge-m3`,
  agent đọc biến này qua `app/embeddings.py`).
- **Nghị định 13/2023/NĐ-CP:** memory cá nhân là dữ liệu cá nhân. User phải có
  quyền xem và xoá, và dữ liệu cần được mã hoá. Xem phần giới hạn bên dưới.

---

## 5. POC này chưa xử lý được

- **Lọc theo ngưỡng liên quan:** RRF luôn trả về đủ top-3. Ở query 5, vị trí 2–3
  là memory không liên quan (Postgres, phở). Cần ngưỡng điểm tối thiểu, và
  ngưỡng chỉ có ý nghĩa khi đổi sang embedding đa ngữ.
- **Push API thật:** hoạt động gần đây đang là buffer trong bộ nhớ của tiến trình.
  Bản thật cần `PushSource` cho `query_velocity_features`.
- **CRUD và quyền được quên:** chưa có `forget(memory_id)` hay xoá theo user.
- **Bảo mật:** chưa mã hoá dữ liệu lưu trữ, chưa kiểm tra quyền ở tầng API
  (`user_id` đang được tin tưởng từ phía gọi).
- **Memory decay và hợp nhất:** memory cũ không bao giờ hết hạn hoặc được gộp.
- **Lưu trữ lâu dài:** Qdrant đang chạy in-memory, mất dữ liệu khi tắt chương trình.

---

## Vibe-coding log

Toàn bộ code và bản nháp tài liệu được viết cùng Claude Code. Bốn quyết định (chunk
~200 token, tabular features, freshness phân tầng, bỏ dấu + tách khoảng trắng)
do tôi chọn sau khi xem các phương án và đánh đổi. **Prompt hiệu quả nhất:** yêu
cầu in thứ hạng của *từng* retriever (BM25, vector) cho 5 query. Nhờ đó thấy
ngay hai lỗi: memory có điểm BM25 = 0 vẫn được cộng điểm RRF, và danh sách profile
ở trọng số 1.0 lấn át cả hai retriever. **Điều chưa ổn:** bản đầu tiên chạy
`exit 0` và trông "đúng", nhưng 5 query cho gần như cùng một kết quả. Code chạy
được chưa có nghĩa là thiết kế đúng; phải đọc kỹ output mới phát hiện.
