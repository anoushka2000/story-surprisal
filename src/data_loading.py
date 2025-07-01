import gutenbergpy.textget

def get_text(id_num):
    # This gets a book by its gutenberg id number
    raw_book = gutenbergpy.textget.get_text_by_id(id_num) # with headers
    return gutenbergpy.textget.strip_headers(raw_book) # without headers