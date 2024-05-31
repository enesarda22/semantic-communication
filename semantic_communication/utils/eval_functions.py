from torch.nn import functional as F
import re
import matplotlib.pyplot as plt


def plotter(x_axis_values, y_axis_values, x_label, y_label, title):
    plt.figure()
    plt.plot(x_axis_values, y_axis_values)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=400)


def sbert_semantic_similarity_score(target_sentence, received_sentence, sbert_model):
    target_emb = sbert_model.encode(target_sentence, convert_to_tensor=True).unsqueeze(0)
    received_emb = sbert_model.encode(received_sentence, convert_to_tensor=True).unsqueeze(0)
    scores = F.cosine_similarity(target_emb, received_emb)
    return scores[0].item()


def semantic_similarity_score(target_sentences, received_sentences, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are skilled in evaluating how similar the two sentences are. Provide a number between -1 "
                "and 1 denoting the semantic similarity score for given sentences A and B with precision "
                "0.01. 1 means they are perfectly similar and -1 means they are opposite while 0 means their "
                "meanings are uncorrelated. Just provide a score without any words or symbols.",
            },
            {
                "role": "user",
                "content": f"A=({target_sentences})  B=({received_sentences})",
            },
        ],
    )

    if completion.choices[0].finish_reason == "stop":
        pattern = re.compile(r"(?<![\d.-])-?(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")
        res = pattern.findall(completion.choices[0].message.content)
        if len(res) == 1:
            return float(res[0])
        else:
            print(res)
            return float("nan")
    else:
        return float("nan")
