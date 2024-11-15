"""add token in ObservationTable 

Revision ID: ce8584f772ae
Revises: 6eb7dd60e642
Create Date: 2024-11-15 11:04:17.042976

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ce8584f772ae'
down_revision: Union[str, None] = '6eb7dd60e642'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('observation_table', sa.Column('nb_tokens_prompt', sa.Integer(), nullable=True))
    op.add_column('observation_table', sa.Column('nb_tokens_completion', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('observation_table', 'nb_tokens_completion')
    op.drop_column('observation_table', 'nb_tokens_prompt')
    # ### end Alembic commands ###
