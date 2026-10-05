# Reflection — Lab 19

**Tên:** Lê Tuấn Anh
**MSSV:** 2A202602952
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 (kw / sem / hyb): tổng **77.8% / 73.2% / 78.6%**.

- **`exact`** (96.7 / 88.7 / 96.7): BM25 thắng, hybrid ngang bằng. Query chứa
  thuật ngữ nguyên văn nên khớp từ là tín hiệu mạnh nhất.
- **`paraphrase`** (33.3 / 24.0 / 32.0): vector lẽ ra phải thắng nhưng lại thấp
  nhất, vì `bge-small-en` huấn luyện trên tiếng Anh, không hiểu câu tiếng Việt
  diễn đạt lại. Cần model đa ngữ như `bge-m3`.
- **`mixed`** (97.0 / 98.5 / **100**): hybrid thắng. RRF thưởng doc đứng cao ở
  *cả hai* danh sách, nên bù được điểm yếu của từng retriever.

**Khi không dùng hybrid:**
- **Pure BM25**: tra mã lỗi, SKU, tên hàm, ID, hoặc khi embedding yếu với ngôn
  ngữ của corpus (như lab này). Hybrid chỉ hơn BM25 0.8 điểm nhưng P99 tăng từ
  2.5 ms lên 22.4 ms và tốn thêm chi phí embed.
- **Pure vector**: query ngôn ngữ tự nhiên, cross-lingual hoặc đa phương thức,
  ít từ chung với tài liệu, và có embedding model đủ mạnh.

---

## Điều ngạc nhiên nhất khi làm lab này

PIT join ban đầu chỉ trả về 2/3 dòng: `u_001` bị hỏi tại thời điểm *trước* khi
feature của nó được ghi, nên Feast từ chối dùng giá trị "tương lai". Đó không
phải lỗi mà chính là cơ chế chống leakage đang hoạt động.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`) — `ARCHITECTURE.md`, `agent.py`, `demo.py`
- [ ] Pair work với: _(làm một mình)_
