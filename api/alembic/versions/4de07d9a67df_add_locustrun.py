"""Add LocustRun

Revision ID: 4de07d9a67df
Revises: f4342e16e891
Create Date: 2025-01-17 00:37:14.821839

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4de07d9a67df'
down_revision: Union[str, None] = 'f4342e16e891'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('locustrun',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('scenario', sa.Text(), nullable=True),
    sa.Column('model', sa.Text(), nullable=True),
    sa.Column('api_url', sa.Text(), nullable=True),
    sa.Column('stats_df', sa.Text(), nullable=True),
    sa.Column('history_df', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('locustrun')
    # ### end Alembic commands ###