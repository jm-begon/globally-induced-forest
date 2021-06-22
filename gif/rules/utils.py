def fstr(v, format="{:.2f}"):
    try:
        if v.size == 1:
            return format.format(v.flatten()[0])
        return format.format(v)
    except AttributeError:
        return format.format(v)
