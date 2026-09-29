from agents.trl import TRLAgent
from agents.gtrl import GTRLAgent
from agents.crl import CRLAgent
from agents.qrl import QRLAgent
from agents.gcfbc import GCFBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.td import TDAgent
from agents.mc import MCAgent

agents = dict(
    trl=TRLAgent,
    gtrl=GTRLAgent,
    crl=CRLAgent,
    qrl=QRLAgent,
    gcfbc=GCFBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    td=TDAgent,
    mc=MCAgent,
)
