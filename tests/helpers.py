"""Helpers that are used across the test suite."""

import coreax.reduction


class CoresetMock(coreax.reduction.Coreset):
    """Test version of :class:`Coreset` with all methods implemented."""

    def fit_to_size(self, coreset_size: int) -> None:
        raise NotImplementedError


class CoresubsetMock(coreax.reduction.Coresubset):
    """Test version of :class:`Coresubset` with all methods implemented."""

    def fit_to_size(self, coreset_size: int) -> None:
        raise NotImplementedError
