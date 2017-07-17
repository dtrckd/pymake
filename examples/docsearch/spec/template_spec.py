from pymake import ExpSpace, ExpTensor, ExpDesign

class docsearch_spec(ExpDesign):

    test_expe = ExpSpace(test=42)

    test_design = ExpTensor(test=[42,43])
