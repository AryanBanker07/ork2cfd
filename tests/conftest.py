"""Shared test fixtures: synthetic OpenRocket rockets and optional local team files."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import pytest

REAL_ORK_CANDIDATES = [
    Path(
        r"C:\Users\banke\OneDrive\Desktop\abhyuday\year 2\prefinal_ork_brahma\brahma_prefinal.ork"
    ),
    Path(r"C:\Users\banke\OneDrive\Desktop\abhyuday\year 2\Brahma\brahma.ork"),
    Path(r"C:\Users\banke\OneDrive\Desktop\abhyuday\year 2\rocket\spaceshot.ork"),
    Path(r"C:\Users\banke\OneDrive\Desktop\abhyuday\year 2\Openrocket Mods\banana3.ork"),
    Path(r"C:\Users\banke\OneDrive\Desktop\abhyuday\year 1\Garuda_1.10.ork\rocket.ork"),
]


def make_synthetic_ork_xml(
    name: str = "TestRocket",
    nose_shape: str = "haack",
    nose_param: float = 0.0,
    fin_tag: str = "trapezoidfinset",
    cross_section: str = "square",
    cant_deg: float = 0.0,
    include_internals: bool = True,
    include_rail_button: bool = False,
) -> str:
    """Create a synthetic OpenRocket 1.10 XML document covering multiple shapes/fins."""
    internals = ""
    if include_internals:
        internals = """
              <bulkhead><name>Bulkhead</name><length>0.01</length><outerradius>auto</outerradius></bulkhead>
              <centeringring><name>CR</name><length>0.01</length></centeringring>
              <parachute><name>Main</name></parachute>
              <masscomponent><name>Payload</name></masscomponent>
        """
    rail_xml = ""
    if include_rail_button:
        rail_xml = """
              <railbutton>
                <name>Rail Button</name>
                <instancecount>2</instancecount>
                <instanceseparation>0.4</instanceseparation>
                <axialoffset method="top">0.2</axialoffset>
                <position type="top">0.2</position>
                <outerdiameter>0.01</outerdiameter>
                <height>0.008</height>
                <angleoffset method="relative">45.0</angleoffset>
              </railbutton>
        """

    if fin_tag == "ellipticalfinset":
        fin_xml = f"""
              <ellipticalfinset>
                <name>Elliptical Fins</name>
                <fincount>4</fincount>
                <position type="bottom">-0.02</position>
                <thickness>0.003</thickness>
                <crosssection>{cross_section}</crosssection>
                <cant>{cant_deg}</cant>
                <rootchord>0.24</rootchord>
                <height>0.11</height>
              </ellipticalfinset>
        """
    elif fin_tag == "freeformfinset":
        fin_xml = f"""
              <freeformfinset>
                <name>Freeform Fins</name>
                <fincount>3</fincount>
                <position type="bottom">0.0</position>
                <thickness>0.003</thickness>
                <crosssection>{cross_section}</crosssection>
                <cant>{cant_deg}</cant>
                <finpoints>
                  <point x="0.0" y="0.0"/>
                  <point x="0.08" y="0.12"/>
                  <point x="0.22" y="0.10"/>
                  <point x="0.25" y="0.0"/>
                </finpoints>
              </freeformfinset>
        """
    else:
        fin_xml = f"""
              <trapezoidfinset>
                <name>Trap Fins</name>
                <fincount>4</fincount>
                <position type="bottom">0.0</position>
                <thickness>0.004</thickness>
                <crosssection>{cross_section}</crosssection>
                <cant>{cant_deg}</cant>
                <rotation>0.0</rotation>
                <filletradius>0.005</filletradius>
                <rootchord>0.25</rootchord>
                <tipchord>0.10</tipchord>
                <sweeplength>0.12</sweeplength>
                <height>0.12</height>
              </trapezoidfinset>
        """

    return f"""<?xml version="1.0" encoding="utf-8"?>
<openrocket version="1.10" creator="OpenRocket 24.12">
  <rocket>
    <name>{name}</name>
    <motorconfiguration configid="cfg1" default="true">
      <stage number="0" active="true"/>
    </motorconfiguration>
    <subcomponents>
      <stage>
        <name>Sustainer</name>
        <subcomponents>
          <nosecone>
            <name>Nose Cone</name>
            <finish>smooth</finish>
            <length>0.45</length>
            <shape>{nose_shape}</shape>
            <shapeparameter>{nose_param}</shapeparameter>
            <shapeclipped>false</shapeclipped>
            <aftradius>auto</aftradius>
          </nosecone>
          <bodytube>
            <name>Upper Tube</name>
            <finish>normal</finish>
            <length>0.60</length>
            <radius>0.05</radius>
            <subcomponents>
              {internals}
              {rail_xml}
            </subcomponents>
          </bodytube>
          <bodytube>
            <name>Lower Tube</name>
            <finish>normal</finish>
            <length>0.80</length>
            <radius>auto 0.05</radius>
            <subcomponents>
              {fin_xml}
            </subcomponents>
          </bodytube>
          <transition>
            <name>Boattail</name>
            <finish>normal</finish>
            <length>0.10</length>
            <shape>conical</shape>
            <foreradius>auto</foreradius>
            <aftradius>0.035</aftradius>
          </transition>
        </subcomponents>
      </stage>
    </subcomponents>
  </rocket>
</openrocket>
"""


@pytest.fixture
def synthetic_ork_file(tmp_path: Path) -> Path:
    """Plain XML .ork file."""
    p = tmp_path / "synthetic_rocket.ork"
    p.write_text(make_synthetic_ork_xml(include_rail_button=True), encoding="utf-8")
    return p


@pytest.fixture
def synthetic_zip_ork_file(tmp_path: Path) -> Path:
    """ZIP-compressed .ork file."""
    p = tmp_path / "synthetic_zip.ork"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("rocket.ork", make_synthetic_ork_xml(name="ZipRocket"))
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def synthetic_gzip_ork_file(tmp_path: Path) -> Path:
    """GZIP-compressed .ork file."""
    p = tmp_path / "synthetic_gz.ork"
    p.write_bytes(gzip.compress(make_synthetic_ork_xml(name="GzRocket").encode("utf-8")))
    return p
