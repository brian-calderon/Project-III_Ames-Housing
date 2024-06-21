from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data.loader import DataSchema
from .loader import DataSchema


# @dataclass(slots=True)
@dataclass()
class DataSource:
    _data: pd.DataFrame

    def filter(
        self,
        years: Optional[list[str]] = None,
        months: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> DataSource:
        if years is None:
            years = self.unique_years
        if months is None:
            months = self.unique_months
        if categories is None:
            categories = self.unique_categories
        # This is "python magic" since we are accessing the years input argument via
        # '@years', which is not really an explicit usage of the kwarg. So it's greyed out
        # in our IDE since the IDE thinks we are not using it, but we are...again python magic...
        # NOTE: The .query() method works like this: df.query('column in @list[str]'), first you
        # give it the column you want to filter and then the list against you want to filer.
        filtered_data = self._data.query(
            "year in @years and month in @months and category in @categories"
        )
        return DataSource(filtered_data)

    def create_pivot_table(self) -> pd.DataFrame:
        pt = self._data.pivot_table(
            values=DataSchema.AMOUNT,
            index=[DataSchema.CATEGORY],
            aggfunc="sum",
            fill_value=0,
            dropna=False,
        )
        return pt.reset_index().sort_values(DataSchema.AMOUNT, ascending=False)

    @property
    def row_count(self) -> int:
        return self._data.shape[0]

    @property
    def all_years(self) -> list[str]:
        return self._data[DataSchema.YEAR].tolist()

    @property
    def all_months(self) -> list[str]:
        return self._data[DataSchema.MONTH].tolist()

    @property
    def all_categories(self) -> list[str]:
        return self._data[DataSchema.CATEGORY].tolist()

    @property
    def all_amounts(self) -> list[str]:
        return self._data[DataSchema.AMOUNT].tolist()

    @property
    def unique_years(self) -> list[str]:
        # NOTE: set() automatically removes duplicates but makes it an iterator. In this case
        # you don't care that it's an iterator as long it displays unique values for years.
        return sorted(set(self.all_years), key=int)

    @property
    def unique_months(self) -> list[str]:
        # NOTE: set() automatically removes duplicates but makes it an iterator. In this case
        # you don't care that it's an iterator as long it displays unique values for months.
        return sorted(set(self.all_months))

    @property
    def unique_categories(self) -> list[str]:
        # NOTE: set() automatically removes duplicates but makes it an iterator. In this case
        # you don't care that it's an iterator as long it displays unique values for categories.
        return sorted(set(self.all_categories))