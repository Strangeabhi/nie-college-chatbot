import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load FAQ data
def load_faq_data(path='faq_data.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions, answers = [], []
    for cat in data:
        for qa in cat['questions']:
            questions.append(qa['question'])
            answers.append(qa['answer'])
    return questions, answers

# Load embeddings
def load_embeddings(path='faq_embeddings.npy'):
    import numpy as np
    return np.load(path)

def chatbot_answer(question, questions_list, answers_list, embeddings, model, threshold=0.8):
    query_emb = model.encode([question])
    # Normalize embeddings for cosine similarity
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_emb / np.linalg.norm(query_emb)
    sims = np.dot(emb_norm, query_norm.T).squeeze()
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    if best_score >= threshold:
        return answers_list[best_idx]
    else:
        return ""

if __name__ == '__main__':
    print("🔹 Loading data...")
    questions, expected_answers = load_faq_data()
    embeddings_data = load_embeddings()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    correct = 0
    failed = []

    print("🔹 Testing accuracy...")
    for q, expected in zip(questions, expected_answers):
        actual = chatbot_answer(q, questions, expected_answers, embeddings_data, model)
        emb_expected = model.encode(expected, convert_to_tensor=True)
        emb_actual = model.encode(actual, convert_to_tensor=True)
        sim = util.cos_sim(emb_expected, emb_actual).item()
        
        if sim >= 0.8:
            correct += 1
        else:
            failed.append({
                'question': q,
                'expected': expected,
                'actual': actual,
                'similarity': sim
            })

    total = len(questions)
    accuracy = correct / total if total else 0
    print(f"\n✅ Final Accuracy: {accuracy:.2%} ({correct}/{total})")

    with open('accuracy_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Total questions tested: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {total - correct}\n\n")
        if failed:
            f.write("Failed questions:\n")
            for fail in failed:
                f.write(f"Q: {fail['question']}\n")
                f.write(f"Expected: {fail['expected']}\n")
                f.write(f"Actual: {fail['actual']}\n")
                f.write(f"Similarity: {fail['similarity']:.3f}\n\n")
        else:
            f.write("All questions answered correctly!\n")

    print("\n📄 Detailed report written to accuracy_report.txt")
