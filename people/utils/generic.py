
def appstr(s, a):
    """Safe appending strings."""
    try:
        return s + a
    except:
        return None

