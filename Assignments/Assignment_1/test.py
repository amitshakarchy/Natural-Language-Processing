import ex1

text = "A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat " \
       "on the rat that sat on the cat on the mat. "

nt = ex1.normalize_text(text)  # lower casing, padding punctuation with white spaces
print(nt)

for n in [
    3,
    2,
    5,
    8,
    0,
    1,
]:
    print("n:", n)
    lm = ex1.Ngram_Language_Model(n=n, chars=False)
    lm.build_model(nt)  # *
    print(lm.get_model_dictionary())  # *
    print()
    for cont in [
        None,
        "a cat",
        "a",
        "non",
        "a cat a cat ac cat a cat a cat ac cat a cat a cat ac cat"]:
        print("cont:", cont)
        t = lm.generate(context=cont, n=10)
        for e in [
            t,
            'a cat sat on the mat . a fat cat sat on the mat . a bat spat on the mat . a rat sat on the mat .',
            'a cat sat on the mat',
            'the rat sat on the cat']:  # *
            print('%s | %.3f' % (e, lm.evaluate(e)))
