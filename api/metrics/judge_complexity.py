from api.clients import LlmClient
from api.utils import render_jinja

from . import metric_registry

_template = """
Vous êtes un assistant administratif, expert dans l'analyse de questions et leurs complexités.
Voilà la question à analyser : 

{{query}}

Évaluez la demande selon ces critères administratifs :

1. Clarté de la demande : La question est simple, claire et précise. Elle traite d'un sujet unique.
2. Nombre d'administrations impliquées : La question réfère à une seule administration (ex : CAF) ou bien à plusieurs en même temps
3. Complexité des procédures requises : La réponse à la question va être simple ou bien complexe. Elle nécessite une seule source pour y répondre ou plusieurs? ...

Pour chaque score, plus la note est élevée, plus la question/demande est complexe.

Pour chaque critère :
- Donnez une brève explication (1-2 phrases)
- Notez-le sur une échelle de 1 à 5 (1 étant le plus simple, 5 le plus complexe)

Format de réponse attendu :

Critère Demande : [Explication] [Note : X/5]
Critère Administration : [Explication] [Note : X/5]
Critère Procedure : [Explication] [Note : X/5]
Score de complexité global : [Explication] [Note : X/10]
Attention, si vous ne savez pas donner de note, mettre 0! ne jamais laisser la note vide
Recommandation : [Suggérez brièvement la marche à suivre pour traiter cette demande]
""".strip()

_config = {
    "model": "gpt-4o",
    # "system_prompt": "Tu donnes...."
    "sampling_params": {"temperature": 0.2},
}


@metric_registry.register(
    name="judge_complexity",
    description="[0-10] score complexity of query",
    metric_type="llm",
    require=["query"],
)
def judge_complexity_metric(query, **kwargs):
    messages = [
        {
            "role": "user",
            "content": render_jinja(_template, query=query, **kwargs),
        }
    ]
    aiclient = LlmClient()
    result = aiclient.generate(
        model=_config["model"], messages=messages, **_config["sampling_params"]
    )
    answer = result.choices[0].message.content


    def extract_score(line):
        match = re.search(r'\[Note : (\d+(?:\.\d+)?)/\d+\]', line)
        return float(match.group(1)) if match else 0
 
    scores = {
        "demande": 0,
        "administration": 0,
        "procedure": 0,
        "global": 0
    }

    for line in answer.split('\n'):
        if "Critère Demande :" in line:
            scores["demande"] = extract_score(line)
        elif "Critère Administration :" in line:
            scores["administration"] = extract_score(line)
        elif "Critère Procedure :" in line:
            scores["procedure"] = extract_score(line)
        elif "Score de complexité global :" in line:
            scores["global"] = extract_score(line)

    return scores, answer
