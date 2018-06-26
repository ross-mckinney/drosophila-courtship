# -*- coding: utf-8 -*-

"""
.. module:: ml
   :synopsis: Generic Errors for ml.

.. moduleauthor:: Ross McKinney
"""


class InvalidParams(Exception):
    def __init__(self, message):
        self.message = message
