"""Pagination helper."""


def paginate(items, page, per_page):
    """Return one page of `items` as a dict.

    `page` is 1-based. The dict has the keys: items (the slice for this page), page, pages (the total number of
    pages, at least 1 even for an empty list), has_prev and has_next (booleans).
    Raises ValueError if per_page < 1 or if page is not in 1..pages.
    """
    if per_page < 1:
        raise ValueError("per_page must be >= 1")
    pages = max(1, len(items) // per_page)
    if page < 1 or page > pages:
        raise ValueError("page out of range")
    start = (page - 1) * per_page
    return {
        "items": list(items[start:start + per_page]),
        "page": page,
        "pages": pages,
        "has_prev": page > 1,
        "has_next": page < pages,
    }
