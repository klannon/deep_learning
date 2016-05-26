The idea: Protocol buffers are a simple way to serialize data so you don't have to write your own serialization format.  First, write a .proto file according to the langauge specification (see below), then you run it through the protocol buffer compiler (protoc) to produce bindings for your programming language of choice, and then you use the bindings you generated to read and write data to files.

Language Specification reference: https://developers.google.com/protocol-buffers/docs/overview

What the compiled bindings look like in Python Reference: https://developers.google.com/protocol-buffers/docs/reference/python-generated
