"""Prints the WDDM video-memory budget (DXGI QueryVideoMemoryInfo) of every adapter for this process.

On Windows the driver silently backs device allocations beyond the local budget with *system*
memory (read over PCIe at ~28 GB/s), while hipMemGetInfo still reports them as VRAM.

    python -I tools/dxgi_budget.py
"""

import ctypes
from ctypes import wintypes

GUID = ctypes.c_byte * 16


def guid(s):
    import uuid
    return GUID.from_buffer_copy(uuid.UUID(s).bytes_le)


IID_IDXGIFactory1 = guid("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIAdapter3 = guid("645967A4-1392-4310-A798-8053CE3E93FD")


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", wintypes.WCHAR * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_uint64), ("Flags", wintypes.UINT)]


class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]


def vcall(obj, idx, restype, *argtypes):
    vtbl = ctypes.cast(ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[idx])


def main():
    dxgi = ctypes.WinDLL("dxgi.dll")
    fac = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(fac))
    assert hr == 0, hex(hr & 0xFFFFFFFF)
    i = 0
    gib = float(1 << 30)
    while True:
        ad = ctypes.c_void_p()
        hr = vcall(fac, 12, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(fac, i, ctypes.byref(ad))
        if hr != 0:
            break
        desc = DXGI_ADAPTER_DESC1()
        vcall(ad, 10, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC1))(ad, ctypes.byref(desc))
        a3 = ctypes.c_void_p()
        if vcall(ad, 0, ctypes.c_long, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(
                ad, ctypes.byref(IID_IDXGIAdapter3), ctypes.byref(a3)) == 0:
            for grp, name in [(0, "local"), (1, "non-local")]:
                info = DXGI_QUERY_VIDEO_MEMORY_INFO()
                vcall(a3, 14, ctypes.c_long, wintypes.UINT, ctypes.c_int, ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO))(
                    a3, 0, grp, ctypes.byref(info))
                print(f"[{i}] {desc.Description} ({desc.DedicatedVideoMemory / gib:.2f} GiB dedicated) {name}: "
                      f"budget {info.Budget / gib:.2f} GiB, usage {info.CurrentUsage / gib:.2f}, "
                      f"available-for-reservation {info.AvailableForReservation / gib:.2f}, "
                      f"reserved {info.CurrentReservation / gib:.2f}")
        i += 1


if __name__ == "__main__":
    main()
