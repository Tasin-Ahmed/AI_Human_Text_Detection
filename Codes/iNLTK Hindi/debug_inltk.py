try:
    from inltk.inltk import tokenize, setup
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
