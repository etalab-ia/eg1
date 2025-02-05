"""Mandatory readmes

Revision ID: 114ab6896a16
Revises: 1972b3f9769f
Create Date: 2024-12-14 02:10:57.216884

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '114ab6896a16'
down_revision: Union[str, None] = '1972b3f9769f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('datasets', sa.Column('readme', sa.Text(), nullable=True))
    # ### end Alembic commands ###

     # Add default Readmes...
    op.execute(
        sa.text("UPDATE datasets SET readme = 'nr' WHERE readme IS NULL")
    )
    op.execute(
        sa.text("UPDATE experiment_sets SET readme = 'nr' WHERE readme IS NULL")
    )

def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('datasets', 'readme')
    # ### end Alembic commands ###
