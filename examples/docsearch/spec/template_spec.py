from pymake import ExpSpace, ExpTensor, ExpDesign, ExpGroup

class docsearch_spec(ExpDesign):

    test_expe = ExpSpace(test=42)

    test_design = ExpTensor(test=[42,43])

    test_group = ExpGroup([test_expe, test_design])
