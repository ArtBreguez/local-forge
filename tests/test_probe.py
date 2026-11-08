import unittest
from unittest import mock

from forge import probe, util

from .helpers import fixture, profile


class ListDevices(unittest.TestCase):
    def test_rx6500xt_idle_line_from_the_brief(self):
        ok, gpus = probe.parse_list_devices(fixture("list-devices-rx6500xt-idle.txt"))
        self.assertTrue(ok)
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual((g["backend"], g["device"], g["name"]), ("Vulkan", "Vulkan0", "AMD Radeon RX 6500 XT"))
        self.assertEqual((g["total_mib"], g["free_mib"]), (4080, 3443))

    def test_full_output_with_backend_chatter_and_less_free_vram(self):
        ok, gpus = probe.parse_list_devices(fixture("list-devices-rx6500xt-busy.txt"))
        self.assertTrue(ok)
        self.assertEqual([(g["name"], g["total_mib"], g["free_mib"]) for g in gpus], [("AMD Radeon RX 6500 XT", 4080, 3105)])

    def test_cpu_only_output_is_authoritative_and_empty(self):
        ok, gpus = probe.parse_list_devices(fixture("list-devices-cpu-only.txt"))
        self.assertTrue(ok)               # the build ran and said "(none)": do not fall back to OS tools
        self.assertEqual(gpus, [])

    def test_other_backends_and_names_with_parentheses(self):
        text = ("Available devices:\n"
                "  CUDA0: NVIDIA GeForce RTX 4090 (Laptop) (16376 MiB, 15001 MiB free)\n"
                "  ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24380 MiB free)\n"
                "  MTL0: Apple M2 Pro (21845 MiB, 21845 MiB free)\n"
                "  RPC[host:50052]: remote (8000 MiB, 8000 MiB free)\n"
                "  CPU: AMD Ryzen (31000 MiB, 20000 MiB free)\n")
        ok, gpus = probe.parse_list_devices(text)
        self.assertEqual([(g["backend"], g["total_mib"], g["free_mib"]) for g in gpus],
                         [("CUDA", 16376, 15001), ("ROCm", 24560, 24380), ("Metal", 21845, 21845)])
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4090 (Laptop)")

    def test_garbage_is_not_understood(self):
        ok, gpus = probe.parse_list_devices("segmentation fault")
        self.assertFalse(ok)
        self.assertEqual(gpus, [])


class OsTools(unittest.TestCase):
    def test_nvidia_smi(self):
        gpus = probe.parse_nvidia_smi(fixture("nvidia-smi-sample.csv"))
        self.assertEqual([(g["name"], g["total_mib"], g["free_mib"], g["backend"]) for g in gpus],
                         [("NVIDIA GeForce RTX 3060", 12288, 11534, "CUDA"), ("NVIDIA GeForce RTX 3060 Laptop GPU", 6144, 4410, "CUDA")])
        self.assertEqual(probe.parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate with the driver"), [])

    def test_meminfo(self):
        r = probe.parse_meminfo(fixture("linux-vm-meminfo.txt"))
        self.assertEqual(r["total_mib"], 7602940 // 1024)
        self.assertEqual(r["available_mib"], 6332328 // 1024)

    def test_meminfo_without_memavailable(self):
        r = probe.parse_meminfo("MemTotal: 2048000 kB\nMemFree: 100000 kB\nBuffers: 50000 kB\nCached: 150000 kB\n")
        self.assertEqual(r["available_mib"], 300000 // 1024)

    def test_cpuinfo_vm(self):
        r = probe.parse_cpuinfo(fixture("linux-vm-cpuinfo.txt"))
        self.assertEqual(r["model"], "AMD Ryzen 5 3600 6-Core Processor")
        self.assertEqual((r["logical_threads"], r["physical_cores"]), (2, 2))

    def test_cpuinfo_smt_counts_cores_not_threads(self):
        one = "processor : %d\nmodel name : X\nphysical id : 0\ncore id : %d\n\n"
        text = "".join(one % (i, i % 2) for i in range(4))        # 4 threads on 2 cores
        r = probe.parse_cpuinfo(text)
        self.assertEqual((r["logical_threads"], r["physical_cores"]), (4, 2))

    def test_cpuinfo_arm_board_trailer_is_not_a_cpu(self):
        text = "".join("processor : %d\nBogoMIPS : 108.00\n\n" % i for i in range(4)) + "Hardware : BCM2711\nRevision : d03114\nSerial : 10000000\nModel : Raspberry Pi 4\n"
        r = probe.parse_cpuinfo(text)
        self.assertEqual(r["logical_threads"], 4)
        self.assertIsNone(r["physical_cores"])          # no core ids: unknown, not guessed

    def test_windows_cim(self):
        r = probe.parse_windows_cim_processor(fixture("windows-cim-processor.json"))
        self.assertEqual((r["model"], r["physical_cores"], r["logical_threads"]), ("AMD Ryzen 5 3600 6-Core Processor", 6, 12))
        g = probe.parse_windows_cim_video(fixture("windows-cim-videocontroller.json"))
        self.assertEqual((g[0]["name"], g[0]["total_mib"], g[0]["free_mib"], g[0]["backend"]), ("AMD Radeon RX 6500 XT", 4080, None, None))
        # two sockets are summed
        two = '[{"Name":"A","NumberOfCores":8,"NumberOfLogicalProcessors":16},{"Name":"A","NumberOfCores":8,"NumberOfLogicalProcessors":16}]'
        self.assertEqual(probe.parse_windows_cim_processor(two)["physical_cores"], 16)

    def test_power_supply(self):
        on_ac = [{"type": "Mains", "online": "1"}, {"type": "Battery", "status": "Charging", "capacity": "88"}]
        on_batt = [{"type": "Mains", "online": "0"}, {"type": "Battery", "status": "Discharging", "capacity": "61"}]
        desktop = [{"type": "Mains", "online": "1"}]
        mouse = [{"type": "Mains", "online": "1"}, {"type": "Battery", "status": "Discharging", "capacity": "5", "scope": "Device"}]
        self.assertEqual(probe.parse_power_supply(on_ac)["on_battery"], False)
        r = probe.parse_power_supply(on_batt)
        self.assertEqual((r["on_battery"], r["battery_percent"]), (True, 61))
        self.assertEqual(probe.parse_power_supply(desktop)["on_battery"], False)
        self.assertEqual(probe.parse_power_supply(mouse)["on_battery"], False)   # a wireless mouse is not the machine
        self.assertIsNone(probe.parse_power_supply([])["on_battery"])

    def test_pmset(self):
        self.assertTrue(probe.parse_pmset_batt("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1) 54%; discharging")["on_battery"])
        self.assertFalse(probe.parse_pmset_batt("Now drawing from 'AC Power'")["on_battery"])
        self.assertIsNone(probe.parse_pmset_batt("")["on_battery"])


class ProbeAssembly(unittest.TestCase):
    def test_llama_list_devices_is_primary_source(self):
        def fake_run(args, timeout=20, env=None):
            return (0, fixture("list-devices-rx6500xt-busy.txt")) if "--list-devices" in args else None
        with mock.patch.object(util, "find_tool", lambda name, d=None: "/x/" + name), \
                mock.patch.object(util, "run_cmd", fake_run), mock.patch("shutil.which", lambda n: "/usr/bin/nvidia-smi"):
            p = probe.probe()
        self.assertEqual(p["gpu_source"], "llama.cpp")
        self.assertEqual([g["free_mib"] for g in p["gpus"]], [3105])   # nvidia-smi (fake_run -> None) is not consulted

    def test_cpu_only_build_hides_os_level_gpus(self):
        def fake_run(args, timeout=20, env=None):
            return (0, fixture("list-devices-cpu-only.txt")) if "--list-devices" in args else (0, fixture("nvidia-smi-sample.csv"))
        with mock.patch.object(util, "find_tool", lambda name, d=None: "/x/" + name), mock.patch.object(util, "run_cmd", fake_run):
            p = probe.probe()
        self.assertEqual(p["gpus"], [])        # the build cannot use them, so the plan must not count on them

    def test_falls_back_to_os_tools_when_no_llama_binary(self):
        def fake_run(args, timeout=20, env=None):
            return (0, fixture("nvidia-smi-sample.csv")) if "nvidia" in args[0] else None
        with mock.patch.object(util, "find_tool", lambda name, d=None: None), mock.patch.object(util, "run_cmd", fake_run), \
                mock.patch("shutil.which", lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None):
            p = probe.probe()
        self.assertEqual(p["gpu_source"], "os")
        self.assertEqual(p["gpus"][0]["backend"], "CUDA")
        self.assertTrue(any("llama.cpp" in n for n in p["notes"]))

    def test_never_crashes_when_everything_is_missing(self):
        with mock.patch.object(util, "find_tool", lambda name, d=None: None), mock.patch.object(util, "run_cmd", lambda *a, **k: None), \
                mock.patch("shutil.which", lambda n: None), mock.patch("platform.system", lambda: "Plan9"), \
                mock.patch("os.cpu_count", lambda: None), mock.patch("shutil.disk_usage", side_effect=OSError("boom")):
            p = probe.probe(model_dir="/does/not/exist")
        self.assertIsNone(p["cpu"]["physical_cores"])
        self.assertIsNone(p["cpu"]["logical_threads"])
        self.assertIsNone(p["ram"]["total_mib"])
        self.assertEqual(p["gpus"], [])
        self.assertIsNone(p["disk"]["free_mib"])
        self.assertIsNone(p["power"]["on_battery"])
        self.assertTrue(any(n.startswith("disk:") for n in p["notes"]))

    def test_a_crashing_section_is_recorded_not_raised(self):
        with mock.patch.object(probe, "_ram_info", side_effect=RuntimeError("kaput")):
            p = probe.probe()
        self.assertEqual(p["ram"], {"total_mib": None, "available_mib": None})
        self.assertTrue(any("kaput" in n for n in p["notes"]))

    def test_real_probe_on_this_machine_has_the_schema(self):
        p = probe.probe()
        for k in ("os", "cpu", "ram", "gpus", "disk", "power", "fingerprint", "notes"):
            self.assertIn(k, p)
        self.assertGreaterEqual(p["cpu"]["logical_threads"] or 1, 1)


class Fingerprint(unittest.TestCase):
    def test_ignores_load_and_power_but_not_parts(self):
        base = probe.fingerprint(profile())
        self.assertEqual(base, probe.fingerprint(profile(free_vram=900, ram_free=1000, battery=True)))
        self.assertNotEqual(base, probe.fingerprint(profile(gpu=False)))
        self.assertNotEqual(base, probe.fingerprint(profile(cores=8)))
        self.assertNotEqual(base, probe.fingerprint(profile(vram_total=8192)))
        self.assertNotEqual(base, probe.fingerprint(profile(gpu_name="AMD Radeon RX 6600")))


if __name__ == "__main__":
    unittest.main()
