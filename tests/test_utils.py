from unittest import TestCase

from openke.utils import split_nt_line


class TestSplitNTriplesLine(TestCase):
    def setUp(self):
        self.simple_line = "<http://example.org/#spiderman> <http://www.perceive.net/schemas/relationship/enemyOf> " \
                           "<http://example.org/#green-goblin> .\n"
        self.literal_xmls = "<http://example.org/show/218> <http://www.w3.org/2000/01/rdf-schema#label> " \
                            "\"That Seventies Show\"^^<http://www.w3.org/2001/XMLSchema#string> ."
        self.literal_untyped = "<http://example.org/show/218> <http://www.w3.org/2000/01/rdf-schema#label> " \
                               "\"That Seventies Show\" ."
        self.literal_language = "<http://example.org/show/218> <http://example.org/show/localName> " \
                                "\"That Seventies Show\"@en ."
        self.literal_region = "<http://example.org/show/218> <http://example.org/show/localName> " \
                              "\"Cette Série des Années Septante\"@fr-be ."
        self.literal_multiline_quotes = "<http://example.org/#spiderman> <http://example.org/text> " \
                                        "\"This is a multi-line\\nliteral with many quotes (\\\"\\\"\\\"\\\"\\\")" \
                                        "\\nand two apostrophes ('').\" ."
        self.literal_integer = "<http://en.wikipedia.org/wiki/Helium> <http://example.org/elements/atomicNumber> " \
                               "\"2\"^^<http://www.w3.org/2001/XMLSchema#integer> ."
        self.literal_double = "<http://en.wikipedia.org/wiki/Helium> <http://example.org/elements/specificGravity> " \
                              "\"1.663E-4\"^^<http://www.w3.org/2001/XMLSchema#double> ."

    def test_split_simple(self):
        s, p, o = split_nt_line(self.simple_line)
        self.assertEqual(s, "http://example.org/#spiderman")
        self.assertEqual(p, "http://www.perceive.net/schemas/relationship/enemyOf")
        self.assertEqual(o, "http://example.org/#green-goblin")

    def test_literal_xmls(self):
        s, p, o = split_nt_line(self.literal_xmls)
        self.assertEqual(s, "http://example.org/show/218")
        self.assertEqual(p, "http://www.w3.org/2000/01/rdf-schema#label")
        self.assertEqual(o, """"That Seventies Show"^^<http://www.w3.org/2001/XMLSchema#string>""")

    def test_literal_untyped(self):
        s, p, o = split_nt_line(self.literal_untyped)
        self.assertEqual(s, "http://example.org/show/218")
        self.assertEqual(p, "http://www.w3.org/2000/01/rdf-schema#label")
        self.assertEqual(o, "\"That Seventies Show\"")

    def test_literal_language(self):
        s, p, o = split_nt_line(self.literal_language)
        self.assertEqual(s, "http://example.org/show/218")
        self.assertEqual(p, "http://example.org/show/localName")
        self.assertEqual(o, "\"That Seventies Show\"@en")

    def test_literal_region(self):
        s, p, o = split_nt_line(self.literal_region)
        self.assertEqual(s, "http://example.org/show/218")
        self.assertEqual(p, "http://example.org/show/localName")
        self.assertEqual(o, "\"Cette Série des Années Septante\"@fr-be")

    def test_literal_multiline_quotes(self):
        s, p, o = split_nt_line(self.literal_multiline_quotes)
        self.assertEqual(s, "http://example.org/#spiderman")
        self.assertEqual(p, "http://example.org/text")
        self.assertEqual(o, "\"This is a multi-line\\nliteral with many quotes (\\\"\\\"\\\"\\\"\\\")"
                            "\\nand two apostrophes ('').\"")

    def test_literal_integer(self):
        s, p, o = split_nt_line(self.literal_integer)
        self.assertEqual(s, "http://en.wikipedia.org/wiki/Helium")
        self.assertEqual(p, "http://example.org/elements/atomicNumber")
        self.assertEqual(o, "\"2\"^^<http://www.w3.org/2001/XMLSchema#integer>")

    def test_literal_double(self):
        s, p, o = split_nt_line(self.literal_double)
        self.assertEqual(s, "http://en.wikipedia.org/wiki/Helium")
        self.assertEqual(p, "http://example.org/elements/specificGravity")
        self.assertEqual(o, "\"1.663E-4\"^^<http://www.w3.org/2001/XMLSchema#double>")
