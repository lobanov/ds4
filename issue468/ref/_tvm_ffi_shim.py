"""Shim: patch known tvm_ffi / TVM Python interop bugs on this stack.

tvm_ffi 0.1.12 (pulled in by tilelang 0.1.8) iterates a type's FFI schema fields
and setattr's each as a property on the Python class. TVM's ir.DictAttrs type
exposes a field literally named '__dict__'; setattr(cls, '__dict__', ...) then
raises "attribute '__dict__' of 'type' objects is not writable" on CPython
3.11/3.12. We skip that one name.

Self-disables when the target symbols are absent (e.g. on apache-tvm-ffi 0.1.2,
which does not have the bug and where the references below may not exist).
Must be imported BEFORE tilelang/tvm if active. Importing is always safe.
"""
try:
    import tvm_ffi.registry as _r
    _install = getattr(_r, "_install_ffi_init_attr", None)
    _core = getattr(_r, "core", None)
    if (getattr(_r, "_add_class_attrs", None) is None
            or _install is None or _core is None):
        raise ImportError("no-op: tvm_ffi has no patchable symbols (likely pre-bug)")

    def _add_class_attrs_safe(type_cls, type_info):
        for field in type_info.fields:
            name = field.name
            if name == "__dict__":  # reserved; setattr fails on 'type'
                continue
            if name not in type_cls.__dict__:
                setattr(type_cls, name, field.as_property(type_cls))
        has_ffi_init = False
        for method in type_info.methods:
            name = method.name
            if name == "__ffi_init__":
                _r._install_ffi_init_attr(type_cls, type_info, method.func)
                has_ffi_init = True
                continue
            if not hasattr(type_cls, name):
                setattr(type_cls, name, method.as_callable(type_cls))
        if not has_ffi_init:
            ffi_init = _r.core._lookup_type_attr(type_info.type_index, "__ffi_init__")
            if ffi_init is not None:
                _r._install_ffi_init_attr(type_cls, type_info, ffi_init)
        return type_cls

    _r._add_class_attrs = _add_class_attrs_safe
except ImportError:
    pass

try:
    import tvm.runtime.support as _support

    _derived_object = getattr(_support, "derived_object", None)
    if _derived_object is None:
        raise ImportError("no-op: tvm.runtime.support.derived_object missing")

    def _derived_object_safe(cls):
        derived_cls = _derived_object(cls)
        original_setattr = derived_cls.__setattr__

        def __setattr__(self, name, value):
            # tilelang's generated TVMDerivedObject wrappers try to proxy every
            # attribute assignment through self._inst, but some wrapped classes
            # assign fields in __init__ before _inst exists. Guard that path.
            if name not in ["_inst", "key", "handle"] and not hasattr(self, "_inst"):
                super(derived_cls, self).__setattr__(name, value)
                return
            return original_setattr(self, name, value)

        derived_cls.__setattr__ = __setattr__
        return derived_cls

    _support.derived_object = _derived_object_safe
except ImportError:
    pass
