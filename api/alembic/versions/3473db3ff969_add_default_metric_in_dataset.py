"""add default_metric in dataset

Revision ID: 3473db3ff969
Revises: 2affae2e4758
Create Date: 2025-01-31 11:19:43.193068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3473db3ff969'
down_revision: Union[str, None] = '2affae2e4758'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('datasets', sa.Column('default_metric', sa.Text(), nullable=True))
    # ### end Alembic commands ###
    # Update existing rows to set the default_metric 
    op.execute(
        sa.text("UPDATE datasets SET default_metric = 'judge_notator' WHERE default_metric IS NULL")
    )

def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('datasets', 'default_metric')
    # ### end Alembic commands ###
