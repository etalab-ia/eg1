from api.clients import LlmClient
from api.utils import render_jinja

from . import metric_registry

_template = """
Analysez la question, la réponse standard et la réponse d'un autre agent fournie ci-dessous. Votre tâche est d'évaluer les DIVAGATIONS dans la réponse de l'autre agent, c'est-à-dire le contenu non pertinent ou hors sujet.

Question posée :

<A>
{{query}}
</A>


Réponse standard (LA REFERENCE!) :

<B>
{{output_true}}
</B>


Réponse générée par un autre agent à évaluer :

<C>
{{output}}
</C>


Instructions :
    1. Identifiez le contenu de la réponse de l'autre agent qui n'est pas directement lié à la question ou qui n'apporte pas d'information essentielle.
    2. Évaluez la proportion de ce contenu non pertinent dans la réponse de l'autre agent.
    3. Attribuez une note de 1 à 10 pour les divagations :
    - 1 : Aucune divagation, réponse entièrement pertinente
    - 10 : Majoritairement des divagations, peu de contenu pertinent

Répondez UNIQUEMENT avec la note (ET RIEN D AUTRE) sous ce format :
NOTE
""".strip()

_config = {
    "model": "gpt-4o",
    # "system_prompt": "Tu donnes...."
    "sampling_params": {"temperature": 0.2},
}


@metric_registry.register(
    name="judge_rambling",
    description="[1-10] rambling score between output and output_true",
    metric_type="llm",
    require=["output", "output_true", "query"],
)
def judge_pertinence_metric(output, output_true, **kwargs):
    config = _config | {k: v for k, v in kwargs.items() if k in _config}
    messages = [
        {
            "role": "user",
            "content": render_jinja(_template, output=output, output_true=output_true, **kwargs),
        }
    ]
    aiclient = LlmClient()
    result = aiclient.generate(
        model=config["model"], messages=messages, **config["sampling_params"]
    )
    answer = result.choices[0].message.content
    score = answer.strip(" \n\"'.%")
    try:
        score = float(score)
    except ValueError:
        score = None
    return score, answer
