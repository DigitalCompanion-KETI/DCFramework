# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gateway.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gateway.proto',
  package='pb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rgateway.proto\x12\x02pb\"\x07\n\x05\x45mpty\"\x16\n\x07Message\x12\x0b\n\x03Msg\x18\x01 \x01(\t\"\x1a\n\x08Messages\x12\x0e\n\x06Output\x18\x01 \x01(\x0c\"\x87\x04\n\x15\x43reateFunctionRequest\x12\x0f\n\x07Service\x18\x01 \x01(\t\x12\r\n\x05Image\x18\x02 \x01(\t\x12\x37\n\x07\x45nvVars\x18\x03 \x03(\x0b\x32&.pb.CreateFunctionRequest.EnvVarsEntry\x12\x35\n\x06Labels\x18\x04 \x03(\x0b\x32%.pb.CreateFunctionRequest.LabelsEntry\x12?\n\x0b\x41nnotations\x18\x05 \x03(\x0b\x32*.pb.CreateFunctionRequest.AnnotationsEntry\x12\x13\n\x0b\x43onstraints\x18\x06 \x03(\t\x12\x0f\n\x07Secrets\x18\x07 \x03(\t\x12\x14\n\x0cRegistryAuth\x18\x08 \x01(\t\x12%\n\x06Limits\x18\t \x01(\x0b\x32\x15.pb.FunctionResources\x12\'\n\x08Requests\x18\n \x01(\x0b\x32\x15.pb.FunctionResources\x1a.\n\x0c\x45nvVarsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a-\n\x0bLabelsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x32\n\x10\x41nnotationsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"-\n\x15\x44\x65leteFunctionRequest\x12\x14\n\x0c\x46unctionName\x18\x01 \x01(\t\"<\n\x13ScaleServiceRequest\x12\x13\n\x0bServiceName\x18\x01 \x01(\t\x12\x10\n\x08Replicas\x18\x02 \x01(\x04\"6\n\x14InvokeServiceRequest\x12\x0f\n\x07Service\x18\x01 \x01(\t\x12\r\n\x05Input\x18\x02 \x01(\x0c\"\'\n\x0f\x46unctionRequest\x12\x14\n\x0c\x46unctionName\x18\x01 \x01(\t\",\n\tFunctions\x12\x1f\n\tFunctions\x18\x01 \x03(\x0b\x32\x0c.pb.Function\"\xae\x02\n\x08\x46unction\x12\x0c\n\x04Name\x18\x01 \x01(\t\x12\r\n\x05Image\x18\x02 \x01(\t\x12\x17\n\x0fInvocationCount\x18\x03 \x01(\x04\x12\x10\n\x08Replicas\x18\x04 \x01(\x04\x12\x19\n\x11\x41vailableReplicas\x18\x05 \x01(\x04\x12\x32\n\x0b\x41nnotations\x18\x06 \x03(\x0b\x32\x1d.pb.Function.AnnotationsEntry\x12(\n\x06Labels\x18\x07 \x03(\x0b\x32\x18.pb.Function.LabelsEntry\x1a\x32\n\x10\x41nnotationsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a-\n\x0bLabelsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"=\n\x11\x46unctionResources\x12\x0e\n\x06Memory\x18\x01 \x01(\t\x12\x0b\n\x03\x43PU\x18\x02 \x01(\t\x12\x0b\n\x03GPU\x18\x03 \x01(\t2\x97\x04\n\x07Gateway\x12\x31\n\x06Invoke\x12\x18.pb.InvokeServiceRequest\x1a\x0b.pb.Message\"\x00\x12\"\n\x04List\x12\t.pb.Empty\x1a\r.pb.Functions\"\x00\x12\x32\n\x06\x44\x65ploy\x12\x19.pb.CreateFunctionRequest\x1a\x0b.pb.Message\"\x00\x12\x32\n\x06\x44\x65lete\x12\x19.pb.DeleteFunctionRequest\x1a\x0b.pb.Message\"\x00\x12\x32\n\x06Update\x12\x19.pb.CreateFunctionRequest\x1a\x0b.pb.Message\"\x00\x12.\n\x07GetMeta\x12\x13.pb.FunctionRequest\x1a\x0c.pb.Function\"\x00\x12,\n\x06GetLog\x12\x13.pb.FunctionRequest\x1a\x0b.pb.Message\"\x00\x12\x37\n\rReplicaUpdate\x12\x17.pb.ScaleServiceRequest\x1a\x0b.pb.Message\"\x00\x12 \n\x04Info\x12\t.pb.Empty\x1a\x0b.pb.Message\"\x00\x12\'\n\x0bHealthCheck\x12\t.pb.Empty\x1a\x0b.pb.Message\"\x00\x12\x37\n\x07Invokes\x12\x18.pb.InvokeServiceRequest\x1a\x0c.pb.Messages\"\x00(\x01\x30\x01\x62\x06proto3')
)




_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='pb.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=28,
)


_MESSAGE = _descriptor.Descriptor(
  name='Message',
  full_name='pb.Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Msg', full_name='pb.Message.Msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=52,
)


_MESSAGES = _descriptor.Descriptor(
  name='Messages',
  full_name='pb.Messages',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Output', full_name='pb.Messages.Output', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=80,
)


_CREATEFUNCTIONREQUEST_ENVVARSENTRY = _descriptor.Descriptor(
  name='EnvVarsEntry',
  full_name='pb.CreateFunctionRequest.EnvVarsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='pb.CreateFunctionRequest.EnvVarsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='pb.CreateFunctionRequest.EnvVarsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=457,
  serialized_end=503,
)

_CREATEFUNCTIONREQUEST_LABELSENTRY = _descriptor.Descriptor(
  name='LabelsEntry',
  full_name='pb.CreateFunctionRequest.LabelsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='pb.CreateFunctionRequest.LabelsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='pb.CreateFunctionRequest.LabelsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=505,
  serialized_end=550,
)

_CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY = _descriptor.Descriptor(
  name='AnnotationsEntry',
  full_name='pb.CreateFunctionRequest.AnnotationsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='pb.CreateFunctionRequest.AnnotationsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='pb.CreateFunctionRequest.AnnotationsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=552,
  serialized_end=602,
)

_CREATEFUNCTIONREQUEST = _descriptor.Descriptor(
  name='CreateFunctionRequest',
  full_name='pb.CreateFunctionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Service', full_name='pb.CreateFunctionRequest.Service', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Image', full_name='pb.CreateFunctionRequest.Image', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='EnvVars', full_name='pb.CreateFunctionRequest.EnvVars', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Labels', full_name='pb.CreateFunctionRequest.Labels', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Annotations', full_name='pb.CreateFunctionRequest.Annotations', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Constraints', full_name='pb.CreateFunctionRequest.Constraints', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Secrets', full_name='pb.CreateFunctionRequest.Secrets', index=6,
      number=7, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='RegistryAuth', full_name='pb.CreateFunctionRequest.RegistryAuth', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Limits', full_name='pb.CreateFunctionRequest.Limits', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Requests', full_name='pb.CreateFunctionRequest.Requests', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CREATEFUNCTIONREQUEST_ENVVARSENTRY, _CREATEFUNCTIONREQUEST_LABELSENTRY, _CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=83,
  serialized_end=602,
)


_DELETEFUNCTIONREQUEST = _descriptor.Descriptor(
  name='DeleteFunctionRequest',
  full_name='pb.DeleteFunctionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='FunctionName', full_name='pb.DeleteFunctionRequest.FunctionName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=604,
  serialized_end=649,
)


_SCALESERVICEREQUEST = _descriptor.Descriptor(
  name='ScaleServiceRequest',
  full_name='pb.ScaleServiceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ServiceName', full_name='pb.ScaleServiceRequest.ServiceName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Replicas', full_name='pb.ScaleServiceRequest.Replicas', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=651,
  serialized_end=711,
)


_INVOKESERVICEREQUEST = _descriptor.Descriptor(
  name='InvokeServiceRequest',
  full_name='pb.InvokeServiceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Service', full_name='pb.InvokeServiceRequest.Service', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Input', full_name='pb.InvokeServiceRequest.Input', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=713,
  serialized_end=767,
)


_FUNCTIONREQUEST = _descriptor.Descriptor(
  name='FunctionRequest',
  full_name='pb.FunctionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='FunctionName', full_name='pb.FunctionRequest.FunctionName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=769,
  serialized_end=808,
)


_FUNCTIONS = _descriptor.Descriptor(
  name='Functions',
  full_name='pb.Functions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Functions', full_name='pb.Functions.Functions', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=810,
  serialized_end=854,
)


_FUNCTION_ANNOTATIONSENTRY = _descriptor.Descriptor(
  name='AnnotationsEntry',
  full_name='pb.Function.AnnotationsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='pb.Function.AnnotationsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='pb.Function.AnnotationsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=552,
  serialized_end=602,
)

_FUNCTION_LABELSENTRY = _descriptor.Descriptor(
  name='LabelsEntry',
  full_name='pb.Function.LabelsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='pb.Function.LabelsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='pb.Function.LabelsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=505,
  serialized_end=550,
)

_FUNCTION = _descriptor.Descriptor(
  name='Function',
  full_name='pb.Function',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Name', full_name='pb.Function.Name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Image', full_name='pb.Function.Image', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='InvocationCount', full_name='pb.Function.InvocationCount', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Replicas', full_name='pb.Function.Replicas', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='AvailableReplicas', full_name='pb.Function.AvailableReplicas', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Annotations', full_name='pb.Function.Annotations', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Labels', full_name='pb.Function.Labels', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FUNCTION_ANNOTATIONSENTRY, _FUNCTION_LABELSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=857,
  serialized_end=1159,
)


_FUNCTIONRESOURCES = _descriptor.Descriptor(
  name='FunctionResources',
  full_name='pb.FunctionResources',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Memory', full_name='pb.FunctionResources.Memory', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='CPU', full_name='pb.FunctionResources.CPU', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='GPU', full_name='pb.FunctionResources.GPU', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1161,
  serialized_end=1222,
)

_CREATEFUNCTIONREQUEST_ENVVARSENTRY.containing_type = _CREATEFUNCTIONREQUEST
_CREATEFUNCTIONREQUEST_LABELSENTRY.containing_type = _CREATEFUNCTIONREQUEST
_CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY.containing_type = _CREATEFUNCTIONREQUEST
_CREATEFUNCTIONREQUEST.fields_by_name['EnvVars'].message_type = _CREATEFUNCTIONREQUEST_ENVVARSENTRY
_CREATEFUNCTIONREQUEST.fields_by_name['Labels'].message_type = _CREATEFUNCTIONREQUEST_LABELSENTRY
_CREATEFUNCTIONREQUEST.fields_by_name['Annotations'].message_type = _CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY
_CREATEFUNCTIONREQUEST.fields_by_name['Limits'].message_type = _FUNCTIONRESOURCES
_CREATEFUNCTIONREQUEST.fields_by_name['Requests'].message_type = _FUNCTIONRESOURCES
_FUNCTIONS.fields_by_name['Functions'].message_type = _FUNCTION
_FUNCTION_ANNOTATIONSENTRY.containing_type = _FUNCTION
_FUNCTION_LABELSENTRY.containing_type = _FUNCTION
_FUNCTION.fields_by_name['Annotations'].message_type = _FUNCTION_ANNOTATIONSENTRY
_FUNCTION.fields_by_name['Labels'].message_type = _FUNCTION_LABELSENTRY
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['Message'] = _MESSAGE
DESCRIPTOR.message_types_by_name['Messages'] = _MESSAGES
DESCRIPTOR.message_types_by_name['CreateFunctionRequest'] = _CREATEFUNCTIONREQUEST
DESCRIPTOR.message_types_by_name['DeleteFunctionRequest'] = _DELETEFUNCTIONREQUEST
DESCRIPTOR.message_types_by_name['ScaleServiceRequest'] = _SCALESERVICEREQUEST
DESCRIPTOR.message_types_by_name['InvokeServiceRequest'] = _INVOKESERVICEREQUEST
DESCRIPTOR.message_types_by_name['FunctionRequest'] = _FUNCTIONREQUEST
DESCRIPTOR.message_types_by_name['Functions'] = _FUNCTIONS
DESCRIPTOR.message_types_by_name['Function'] = _FUNCTION
DESCRIPTOR.message_types_by_name['FunctionResources'] = _FUNCTIONRESOURCES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.Empty)
  })
_sym_db.RegisterMessage(Empty)

Message = _reflection.GeneratedProtocolMessageType('Message', (_message.Message,), {
  'DESCRIPTOR' : _MESSAGE,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.Message)
  })
_sym_db.RegisterMessage(Message)

Messages = _reflection.GeneratedProtocolMessageType('Messages', (_message.Message,), {
  'DESCRIPTOR' : _MESSAGES,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.Messages)
  })
_sym_db.RegisterMessage(Messages)

CreateFunctionRequest = _reflection.GeneratedProtocolMessageType('CreateFunctionRequest', (_message.Message,), {

  'EnvVarsEntry' : _reflection.GeneratedProtocolMessageType('EnvVarsEntry', (_message.Message,), {
    'DESCRIPTOR' : _CREATEFUNCTIONREQUEST_ENVVARSENTRY,
    '__module__' : 'gateway_pb2'
    # @@protoc_insertion_point(class_scope:pb.CreateFunctionRequest.EnvVarsEntry)
    })
  ,

  'LabelsEntry' : _reflection.GeneratedProtocolMessageType('LabelsEntry', (_message.Message,), {
    'DESCRIPTOR' : _CREATEFUNCTIONREQUEST_LABELSENTRY,
    '__module__' : 'gateway_pb2'
    # @@protoc_insertion_point(class_scope:pb.CreateFunctionRequest.LabelsEntry)
    })
  ,

  'AnnotationsEntry' : _reflection.GeneratedProtocolMessageType('AnnotationsEntry', (_message.Message,), {
    'DESCRIPTOR' : _CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY,
    '__module__' : 'gateway_pb2'
    # @@protoc_insertion_point(class_scope:pb.CreateFunctionRequest.AnnotationsEntry)
    })
  ,
  'DESCRIPTOR' : _CREATEFUNCTIONREQUEST,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.CreateFunctionRequest)
  })
_sym_db.RegisterMessage(CreateFunctionRequest)
_sym_db.RegisterMessage(CreateFunctionRequest.EnvVarsEntry)
_sym_db.RegisterMessage(CreateFunctionRequest.LabelsEntry)
_sym_db.RegisterMessage(CreateFunctionRequest.AnnotationsEntry)

DeleteFunctionRequest = _reflection.GeneratedProtocolMessageType('DeleteFunctionRequest', (_message.Message,), {
  'DESCRIPTOR' : _DELETEFUNCTIONREQUEST,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.DeleteFunctionRequest)
  })
_sym_db.RegisterMessage(DeleteFunctionRequest)

ScaleServiceRequest = _reflection.GeneratedProtocolMessageType('ScaleServiceRequest', (_message.Message,), {
  'DESCRIPTOR' : _SCALESERVICEREQUEST,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.ScaleServiceRequest)
  })
_sym_db.RegisterMessage(ScaleServiceRequest)

InvokeServiceRequest = _reflection.GeneratedProtocolMessageType('InvokeServiceRequest', (_message.Message,), {
  'DESCRIPTOR' : _INVOKESERVICEREQUEST,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.InvokeServiceRequest)
  })
_sym_db.RegisterMessage(InvokeServiceRequest)

FunctionRequest = _reflection.GeneratedProtocolMessageType('FunctionRequest', (_message.Message,), {
  'DESCRIPTOR' : _FUNCTIONREQUEST,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.FunctionRequest)
  })
_sym_db.RegisterMessage(FunctionRequest)

Functions = _reflection.GeneratedProtocolMessageType('Functions', (_message.Message,), {
  'DESCRIPTOR' : _FUNCTIONS,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.Functions)
  })
_sym_db.RegisterMessage(Functions)

Function = _reflection.GeneratedProtocolMessageType('Function', (_message.Message,), {

  'AnnotationsEntry' : _reflection.GeneratedProtocolMessageType('AnnotationsEntry', (_message.Message,), {
    'DESCRIPTOR' : _FUNCTION_ANNOTATIONSENTRY,
    '__module__' : 'gateway_pb2'
    # @@protoc_insertion_point(class_scope:pb.Function.AnnotationsEntry)
    })
  ,

  'LabelsEntry' : _reflection.GeneratedProtocolMessageType('LabelsEntry', (_message.Message,), {
    'DESCRIPTOR' : _FUNCTION_LABELSENTRY,
    '__module__' : 'gateway_pb2'
    # @@protoc_insertion_point(class_scope:pb.Function.LabelsEntry)
    })
  ,
  'DESCRIPTOR' : _FUNCTION,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.Function)
  })
_sym_db.RegisterMessage(Function)
_sym_db.RegisterMessage(Function.AnnotationsEntry)
_sym_db.RegisterMessage(Function.LabelsEntry)

FunctionResources = _reflection.GeneratedProtocolMessageType('FunctionResources', (_message.Message,), {
  'DESCRIPTOR' : _FUNCTIONRESOURCES,
  '__module__' : 'gateway_pb2'
  # @@protoc_insertion_point(class_scope:pb.FunctionResources)
  })
_sym_db.RegisterMessage(FunctionResources)


_CREATEFUNCTIONREQUEST_ENVVARSENTRY._options = None
_CREATEFUNCTIONREQUEST_LABELSENTRY._options = None
_CREATEFUNCTIONREQUEST_ANNOTATIONSENTRY._options = None
_FUNCTION_ANNOTATIONSENTRY._options = None
_FUNCTION_LABELSENTRY._options = None

_GATEWAY = _descriptor.ServiceDescriptor(
  name='Gateway',
  full_name='pb.Gateway',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=1225,
  serialized_end=1760,
  methods=[
  _descriptor.MethodDescriptor(
    name='Invoke',
    full_name='pb.Gateway.Invoke',
    index=0,
    containing_service=None,
    input_type=_INVOKESERVICEREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='List',
    full_name='pb.Gateway.List',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_FUNCTIONS,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Deploy',
    full_name='pb.Gateway.Deploy',
    index=2,
    containing_service=None,
    input_type=_CREATEFUNCTIONREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Delete',
    full_name='pb.Gateway.Delete',
    index=3,
    containing_service=None,
    input_type=_DELETEFUNCTIONREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Update',
    full_name='pb.Gateway.Update',
    index=4,
    containing_service=None,
    input_type=_CREATEFUNCTIONREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetMeta',
    full_name='pb.Gateway.GetMeta',
    index=5,
    containing_service=None,
    input_type=_FUNCTIONREQUEST,
    output_type=_FUNCTION,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetLog',
    full_name='pb.Gateway.GetLog',
    index=6,
    containing_service=None,
    input_type=_FUNCTIONREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReplicaUpdate',
    full_name='pb.Gateway.ReplicaUpdate',
    index=7,
    containing_service=None,
    input_type=_SCALESERVICEREQUEST,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Info',
    full_name='pb.Gateway.Info',
    index=8,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='HealthCheck',
    full_name='pb.Gateway.HealthCheck',
    index=9,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_MESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Invokes',
    full_name='pb.Gateway.Invokes',
    index=10,
    containing_service=None,
    input_type=_INVOKESERVICEREQUEST,
    output_type=_MESSAGES,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GATEWAY)

DESCRIPTOR.services_by_name['Gateway'] = _GATEWAY

# @@protoc_insertion_point(module_scope)
