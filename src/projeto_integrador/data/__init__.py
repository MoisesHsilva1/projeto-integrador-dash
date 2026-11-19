"""Camada de acesso a dados: carga, validação de esquema, cache e auditoria."""

from projeto_integrador.data.auditoria import auditar, preparar_transacoes
from projeto_integrador.data.loader import carregar_brutas, carregar_preparadas
from projeto_integrador.data.schema import ErroDeEsquema, validar_esquema

__all__ = [
    "ErroDeEsquema",
    "auditar",
    "carregar_brutas",
    "carregar_preparadas",
    "preparar_transacoes",
    "validar_esquema",
]
