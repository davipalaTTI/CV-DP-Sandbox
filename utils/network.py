import socket
import subprocess
import logging
import psutil
import re
from typing import List, Dict

# ======================== NETWORK UTILITIES ========================

def get_all_local_subnets() -> List[str]:
    """
    Detects ALL active subnets the Jetson is connected to (Wi-Fi, Ethernet, etc.)
    instead of just defaulting to the primary internet connection.
    """
    subnets = []
    try:
        # psutil is already imported at the top of your file!
        for interface_name, interface_addresses in psutil.net_if_addrs().items():
            # Skip loopback and internal docker/virtual interfaces
            if interface_name.startswith('lo') or interface_name.startswith('docker') or interface_name.startswith('veth'):
                continue

            for addr in interface_addresses:
                if addr.family == socket.AF_INET:  # IPv4 only
                    ip = addr.address
                    # Simple subnet calculation (assuming standard /24 networks)
                    subnet = ip.rsplit('.', 1)[0] + '.0/24'
                    if subnet not in subnets:
                        subnets.append(subnet)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to enumerate interfaces: {e}")

    # Fallback to the old dummy-socket method if psutil fails
    if not subnets:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            subnets.append(local_ip.rsplit('.', 1)[0] + '.0/24')
        except Exception:
            subnets.append('192.168.1.0/24')  # Ultimate fallback
        finally:
            s.close()

    return subnets


def get_available_axis_cameras() -> Dict[str, str]:
    """
    Scans ALL local networks for AXIS cameras using nmap and ARP.
    Returns: Dictionary mapping display names to RTSP URLs
    """
    logger = logging.getLogger(__name__)
    subnets = get_all_local_subnets()
    camera_options = {}
    found_ips = []
    axis_prefixes = ['00:40:8c', 'ac:cc:8e', 'b8:a4:4f', 'e8:27:25']

    logger.info(f"Scanning active network interfaces {subnets} for AXIS cameras...")

    # 1. Silent Ping Sweep on ALL found subnets
    for subnet in subnets:
        try:
            subprocess.run(['nmap', '-sn', subnet], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.warning("nmap is not installed. Camera discovery may fail.")
            break  # Stop trying if nmap doesn't exist
        except Exception as e:
            logger.debug(f"Network sweep encountered an issue on {subnet}: {e}")

    # 2. Read Global ARP Table (contains results from all interfaces)
    try:
        arp_output = subprocess.check_output(['arp', '-a'], universal_newlines=True)
        for line in arp_output.splitlines():
            mac_match = re.search(r'(?:[0-9a-fA-F]{1,2}[:-]){5}[0-9a-fA-F]{1,2}', line)
            ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', line)

            if mac_match and ip_match:
                mac_raw = mac_match.group(0).replace('-', ':').lower()
                ip = ip_match.group(0)

                # Standardize MAC format
                mac_parts = [part.zfill(2) for part in mac_raw.split(':')]
                mac_address = ':'.join(mac_parts)

                if any(mac_address.startswith(prefix) for prefix in axis_prefixes):
                    if ip not in found_ips:
                        found_ips.append(ip)
                        logger.debug(f"Discovered AXIS camera at IP: {ip}, MAC: {mac_address}")
    except Exception as e:
        logger.error(f"Failed to read ARP table during camera scan: {e}")

    # 3. Format the Output for the UI
    for index, ip in enumerate(found_ips):
        display_name = f"Axis Camera [{index + 1}] [{ip}]"
        rtsp_url = f"rtsp://root:BWTMSmaster69@{ip}/axis-media/media.amp"
        camera_options[display_name] = rtsp_url

    if found_ips:
        logger.info(f"Found {len(found_ips)} AXIS camera(s) across all networks.")
    else:
        logger.info("No AXIS cameras found on any local network.")

    return camera_options