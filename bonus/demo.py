"""5-query demo for HybridMemoryAgent.  Run:  python bonus/demo.py

Uses Feast data materialized by NB4 (u_001: topic_affinity=cloud). Without it
the demo still runs; the profile line just says it is unavailable.
"""
from __future__ import annotations

from agent import HybridMemoryAgent

MEMORIES_U001 = [
    ("cloud", "Đọc bài về Kubernetes: Deployment quản lý ReplicaSet, còn HPA tự scale số pod "
              "theo CPU. Cần đặt resource requests thì HPA mới tính được phần trăm sử dụng."),
    ("cloud", "Ghi chú: autoscaling hạ tầng theo lưu lượng người dùng — dùng cluster autoscaler "
              "để thêm node khi pod bị pending, scale-in chậm hơn để tránh dao động."),
    ("cloud", "Cloud security checklist: bật MFA cho root account, IAM theo least privilege, "
              "mã hoá S3 bucket at rest, không commit access key vào git."),
    ("security", "Hội thảo bảo mật: Zero Trust nghĩa là không tin mạng nội bộ, mọi request đều "
                 "phải xác thực. Nghị định 13/2023 yêu cầu bảo vệ dữ liệu cá nhân của người dùng VN."),
    ("database", "Postgres: index B-tree cho truy vấn bằng, GIN cho full-text search và JSONB. "
                 "EXPLAIN ANALYZE để xem query plan thật."),
    ("ai_ml", "Học về RAG: chunk tài liệu, embed, lưu vector store, rồi hybrid search BM25 + "
              "vector ghép bằng RRF k=60 trước khi đưa vào LLM."),
    (None, "Công thức phở bò: hầm xương 6 tiếng, nướng gừng và hành tím trước khi cho vào nồi."),
]
SECRET_U002 = "Ghi chú riêng của u_002: mật khẩu wifi nhà là 123456, Kubernetes cluster ở công ty."

QUERIES = [
    ("1. Vector hit",       "Tôi đã đọc gì về Kubernetes?"),
    ("2. Cần profile",      "Recommend đọc gì tiếp"),
    ("3. Activity gần đây", "Tôi đang quan tâm gì gần đây?"),
    ("4. Paraphrase",       "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5. Mixed",            "Cho tôi summary cloud security"),
]


def main() -> None:
    agent = HybridMemoryAgent()
    for topic, text in MEMORIES_U001:
        agent.remember(text, user_id="u_001", topic=topic)
    agent.remember(SECRET_U002, user_id="u_002", topic="cloud")   # must never leak to u_001

    for label, query in QUERIES:
        context = agent.recall(query, user_id="u_001")
        print(f"\n=== {label}: {query!r}\n{context}")
        assert "u_002" not in context, "cross-user leak!"

    # Freshness tier D3: a memory written now is recallable on the very next call.
    agent.remember("Vừa đọc xong: Karpenter thay cluster autoscaler, provision node trong vài giây.",
                   user_id="u_001", topic="cloud")
    print("\n=== Freshness check: 'Karpenter là gì?'")
    print(agent.recall("Karpenter là gì?", user_id="u_001"))


if __name__ == "__main__":
    main()
