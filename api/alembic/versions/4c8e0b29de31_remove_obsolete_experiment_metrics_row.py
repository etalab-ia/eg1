"""Remove obsolete experiment.metrics row

Revision ID: 4c8e0b29de31
Revises: edbdc05e2926
Create Date: 2024-11-22 11:29:06.823741

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4c8e0b29de31'
down_revision: Union[str, None] = 'edbdc05e2926'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('experiments', 'metrics')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('experiments', sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    # ### end Alembic commands ###