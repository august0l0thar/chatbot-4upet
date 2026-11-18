import os
import time

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from supabase import Client, create_client


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

MODELO = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MODELO_EMBEDDING = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
DIMENSAO_EMBEDDING = int(os.getenv("EMBEDDING_DIM", "768"))

LIMIAR_SIMILARIDADE = float(os.getenv("RAG_THRESHOLD", "0.70"))
QUANTIDADE_CHUNKS = int(os.getenv("RAG_MATCH_COUNT", "5"))

MSG_SEM_RESPOSTA = (
    "Não foi possível responder a essa pergunta porque não encontrei "
    "essa informação na base de conhecimento veterinária."
)

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY não configurada.")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL e SUPABASE_KEY precisam estar configuradas.")


# ============================================================================
# CLIENTES
# ============================================================================

client = genai.Client(api_key=GOOGLE_API_KEY)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=MODELO_EMBEDDING,
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=DIMENSAO_EMBEDDING,
)


# ============================================================================
# MODELOS DE RESPOSTA
# ============================================================================

class RespostaRAG(BaseModel):
    contexto_suficiente: bool
    resposta: str
    fontes: list[str]


class ChatbotResponse(BaseModel):
    contexto_suficiente: bool
    resposta: str
    fontes: list[str]


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def gerar_com_retry(tentativas=2, espera=10, **kwargs):
    """
    Repete a chamada apenas quando o Gemini retornar indisponibilidade/503.
    Mantém o mesmo comportamento do notebook original.
    """
    for tentativa in range(1, tentativas + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            erro = str(exc)
            sobrecarga = "503" in erro or "UNAVAILABLE" in erro

            if not sobrecarga or tentativa == tentativas:
                raise

            print(
                f"Servidor Gemini ocupado. "
                f"Nova tentativa em {espera}s..."
            )
            time.sleep(espera)


def resposta_de_erro(exc: Exception) -> ChatbotResponse:
    erro = str(exc)
    print(f"Erro real: {erro[:300]}")

    if "429" in erro or "RESOURCE_EXHAUSTED" in erro:
        mensagem = (
            "[Aviso do Sistema]: Limite de uso da API atingido. "
            "Tente novamente mais tarde."
        )
    elif "503" in erro or "UNAVAILABLE" in erro:
        mensagem = (
            "[Aviso do Sistema]: Os servidores da IA estão sobrecarregados. "
            "Aguarde um pouco e tente novamente."
        )
    else:
        mensagem = "[Aviso do Sistema]: Não foi possível processar a pergunta."

    return ChatbotResponse(
        contexto_suficiente=False,
        resposta=mensagem,
        fontes=[],
    )


def resposta_sem_contexto() -> ChatbotResponse:
    return ChatbotResponse(
        contexto_suficiente=False,
        resposta=MSG_SEM_RESPOSTA,
        fontes=[],
    )


# ============================================================================
# LÓGICA PRINCIPAL DO CHATBOT
# ============================================================================

def consultar_chatbot_veterinario(
    pergunta_usuario: str,
) -> ChatbotResponse:
    pergunta_usuario = pergunta_usuario.strip()

    if not pergunta_usuario:
        return ChatbotResponse(
            contexto_suficiente=False,
            resposta="Digite uma pergunta para o assistente 4UPet.",
            fontes=[],
        )

    print("\n" + "=" * 50)
    print(f"Nova pergunta: {pergunta_usuario}")

    # ------------------------------------------------------------------------
    # 1. EMBEDDING DA PERGUNTA
    # ------------------------------------------------------------------------

    try:
        vetor_pergunta = embeddings.embed_query(
            pergunta_usuario
        )
        print("Embedding gerado.")
    except Exception as exc:
        return resposta_de_erro(exc)

    # ------------------------------------------------------------------------
    # 2. BUSCA VETORIAL NO SUPABASE
    # ------------------------------------------------------------------------

    try:
        resposta_rpc = supabase.rpc(
            "match_rag_documents",
            {
                "query_embedding": vetor_pergunta,
                "match_threshold": LIMIAR_SIMILARIDADE,
                "match_count": QUANTIDADE_CHUNKS,
            },
        ).execute()

        docs = resposta_rpc.data or []

    except Exception as exc:
        print(
            "Erro no Supabase: "
            f"{str(exc)[:300]}"
        )

        return ChatbotResponse(
            contexto_suficiente=False,
            resposta=(
                "[Aviso do Sistema]: Não foi possível acessar "
                "a base de conhecimento."
            ),
            fontes=[],
        )

    print(f"{len(docs)} chunks acima do limiar.")

    for documento in docs:
        metadata = documento.get("metadata") or {}
        print(
            f"  {documento.get('similarity', 0):.3f} | "
            f"{metadata.get('fonte', 'Desconhecida')}"
        )

    # ------------------------------------------------------------------------
    # 3. SEM DOCUMENTOS RELEVANTES
    # ------------------------------------------------------------------------

    if not docs:
        print("Nenhum documento relevante encontrado.")
        return resposta_sem_contexto()

    # ------------------------------------------------------------------------
    # 4. MONTAR CONTEXTO
    # ------------------------------------------------------------------------

    partes_contexto = []

    for documento in docs:
        metadata = documento.get("metadata") or {}
        fonte = metadata.get("fonte", "Desconhecida")
        conteudo = documento.get("content", "")

        partes_contexto.append(
            f"[Fonte: {fonte}]\n{conteudo}\n"
        )

    contexto = "\n".join(partes_contexto)

    prompt = f"""Você é o assistente veterinário 4UPet.

Use APENAS as informações do CONTEXTO abaixo. Não use conhecimento próprio
e não invente fatos.

- contexto_suficiente: true somente se o CONTEXTO responde DIRETAMENTE à
  pergunta. Se ele só trata de assuntos parecidos, marque false.
- Se for true, escreva em "resposta" uma resposta clara em português e
  liste em "fontes" os nomes das fontes usadas.
- Se for false, deixe "resposta" vazia e "fontes" como lista vazia.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta_usuario}"""

    # ------------------------------------------------------------------------
    # 5. GERAÇÃO DA RESPOSTA COM GEMINI
    # ------------------------------------------------------------------------

    try:
        resposta_modelo = gerar_com_retry(
            model=MODELO,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RespostaRAG,
                temperature=0.2,
            ),
        )

        print("Resposta do Gemini recebida.")

    except Exception as exc:
        return resposta_de_erro(exc)

    resultado = resposta_modelo.parsed

    if resultado is None:
        print("Não foi possível interpretar a resposta do modelo.")
        return resposta_sem_contexto()

    # Dependendo da versão do SDK, parsed pode ser Pydantic ou dict.
    if isinstance(resultado, dict):
        resultado = RespostaRAG.model_validate(resultado)

    print(
        "contexto_suficiente="
        f"{resultado.contexto_suficiente}"
    )

    # ------------------------------------------------------------------------
    # 6. RESPOSTA FINAL
    # ------------------------------------------------------------------------

    if (
        resultado.contexto_suficiente
        and resultado.resposta.strip()
    ):
        fontes = list(
            dict.fromkeys(
                fonte.strip()
                for fonte in resultado.fontes
                if fonte and fonte.strip()
            )
        )

        print("Resposta encontrada na base de conhecimento.")

        return ChatbotResponse(
            contexto_suficiente=True,
            resposta=resultado.resposta.strip(),
            fontes=fontes,
        )

    print("O contexto recuperado não responde diretamente à pergunta.")
    return resposta_sem_contexto()
