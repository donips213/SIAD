import streamlit as st
import os as _os
_os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')   # reconstroi .shx ausente/corrompido
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform, transform_bounds
from rasterio.enums import Resampling
import math
from pyproj import Geod
import plotly.graph_objects as go
from skimage.graph import route_through_array
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box, MultiPolygon
from shapely.validation import make_valid
import tempfile
import os
import zipfile
import uuid
import unicodedata
from fpdf import FPDF
import datetime
import time
import matplotlib.pyplot as plt

# --- 0. CONSTANTES NOMEADAS (evita "numeros magicos" espalhados no codigo) ---
CUSTO_AREA_INVALIDA = 10_000_000.0       # penalidade para pixels sem dado de terreno valido
CUSTO_RESTRICAO_SHAPEFILE = 500_000.0    # penalidade por trecho dentro de shapefile de restricao
CUSTO_RESTRICAO_DESENHO_MANUAL = 5_000_000.0  # penalidade por trecho dentro de area desenhada no mapa
CUSTO_PENALIDADE_GRAVIDADE = 1_000_000.0      # penalidade para pontos acima da cota da inicio (rota gravidade)
CUSTO_RESTRICAO_HIDROGRAFIA = 500_000.0       # penalidade por trecho dentro da faixa de rios/corpos d'agua (APP)

G = 9.81                      # m/s2
TOL_VIOLACAO_M = 0.5          # tolerancia (m) para considerar violacao da regra de Slack Flow

def pdf_safe(texto):
    """
    Prepara texto para o FPDF classico (que so aceita latin-1), removendo acentos de forma
    legivel (ex: 'Concentracao' em vez de 'Concentra??o') ao inves do encode('replace') original,
    que trocava qualquer caractere fora do latin-1 por '?' silenciosamente.
    """
    nfkd = unicodedata.normalize('NFKD', str(texto))
    return nfkd.encode('ascii', 'ignore').decode('ascii')

# --- 1. CONFIGURAÇÃO GERAL ---
st.set_page_config(page_title="Master Plan Mineroduto - TCC", layout="wide", page_icon="⛏️")

# --- 2. CLASSE AVANÇADA DE RELATÓRIO (PDF) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'MEMORIAL DE CALCULO E VIABILIDADE TECNICA', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, 'Sistema de Apoio a Decisao para Tracado de Minerodutos (TCC)', 0, 1, 'C')
        self.line(10, 25, 200, 25)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        data = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        self.cell(0, 10, f'Gerado em: {data} | Pagina ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, f"  {label}", 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, pdf_safe(text))
        self.ln()

# --- FUNCOES PURAS DE APOIO (hidraulica/relatorio) ---------------------------
def pressao_admissivel_barlow(od_mm, esp_mm, smys_mpa, fator_projeto):
    """MAOP (MPa) pela formula de Barlow: P = 2 * SMYS * t * F / OD."""
    return 2.0 * smys_mpa * esp_mm * fator_projeto / od_mm


def trechos_criticos(dist_km, margem_m, regra_m, tol_m=TOL_VIOLACAO_M):
    """
    Agrupa amostras consecutivas com margem (HGL - terreno) < regra - tol.
    A tolerancia elimina violacoes espurias de amostragem (ex: 19,95 m para regra de 20 m).
    """
    idx = np.where(margem_m < (regra_m - tol_m))[0]
    if len(idx) == 0:
        return []
    grupos = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [dict(ini=float(dist_km[g[0]]), fim=float(dist_km[g[-1]]),
                 ext=float(dist_km[g[-1]] - dist_km[g[0]]),
                 p_min=float(margem_m[g].min()), n=int(len(g))) for g in grupos]


def analisar_eb2(dist_km, elev, hgl_sem_eb2, regra_m, p_succ_min_m, split_point=None):
    """
    Criterio explicito para a EB-2 (config. so com EB-1):
    - posicao sugerida = ultima amostra a montante da 1a violacao com folga >= regra + pressao minima de succao;
    - carga minima da EB-2 = menor incremento H2 tal que HGL + H2 >= terreno + regra em todo o trecho a jusante
      (inclui o terminal).
    """
    margem = hgl_sem_eb2 - elev
    viol = np.where(margem < regra_m - TOL_VIOLACAO_M)[0]
    out = {'necessaria': bool(len(viol)), 'idx_sugerido': None, 'km_sugerido': None,
           'h2_min_sugerido': None, 'h2_min_usuario': None, 'km_primeira_violacao': None}
    if len(viol):
        i_v = int(viol[0])
        out['km_primeira_violacao'] = float(dist_km[i_v])
        cand = np.where(margem[:i_v] >= regra_m + p_succ_min_m)[0]
        if len(cand):
            s = int(cand[-1])
            out['idx_sugerido'] = s
            out['km_sugerido'] = float(dist_km[s])
            out['h2_min_sugerido'] = float(max(0.0, np.max(elev[s:] + regra_m - hgl_sem_eb2[s:])))
    if split_point is not None:
        out['h2_min_usuario'] = float(max(0.0, np.max(elev[split_point:] + regra_m - hgl_sem_eb2[split_point:])))
    return out


def montar_parametros(inp, res):
    """Tabela de parametros de entrada (reprodutibilidade) - tela e memorial PDF (Tabela 1 do artigo)."""
    mpa_mca = 1.0 / 0.00981
    eb2 = (f"Sim (+{inp['p2_mpa']:.2f} MPa = {inp['p2_mpa']*mpa_mca:.0f} m.c.a)" if inp['eb2'] else "Não")
    return [
        ("Produção anual", f"{inp['prod']:.2f} Mtpa", "Define a vazão do sistema"),
        ("Dias de operação / ano", f"{inp['dias']}", "Eq. 2"),
        ("Disponibilidade física", f"{inp['disp']:.2f}", "Eq. 2"),
        ("Concentração em peso (Cw)", f"{inp['conc']} %", "Sólidos em peso"),
        ("SG do sólido", f"{inp['sg_solido']:.2f}", "Eq. 1"),
        ("d50 do sólido", f"{inp['d50']} um", "Eq. 6 usa d50 em micrometros"),
        ("Diâmetro externo (API 5L)", f"{inp['od_pol']} pol = {inp['od_pol']*25.4:.1f} mm", "Diâmetro NOMINAL externo"),
        ("Espessura de parede", f"{inp['esp_mm']:.2f} mm", ""),
        ("Diâmetro interno usado", f"{res['diam_int_m']:.4f} m", "OD - 2 x espessura"),
        ("Rugosidade absoluta", f"{inp['rug_mm']:.3f} mm", "Eq. 4"),
        ("Viscosidade do carreador", f"{inp['visc_cp']:.2f} cP", "Água"),
        ("Rendimento motor+bomba", f"{inp['eta']:.2f}", "Eq. 12"),
        ("EB-1 (pressão de descarga)", f"{inp['p1_mpa']:.2f} MPa = {inp['p1_mpa']*mpa_mca:.0f} m.c.a = {res['h1']:.0f} m polpa", ""),
        ("EB-2 (incremento de pressão)", eb2, "Posição/carga definidas pelo usuário"),
        ("Regra de Slack Flow", f"{inp['slack']} m", "Folga mínima HGL - terreno"),
        ("Pressão mínima de sucção", f"{inp['p_succ_min_m']} m de polpa", "Critério de posição da EB-2"),
        ("Pressão residual mín. no terminal", f"{inp['p_res_m']} m de polpa", ""),
        ("Aço / fator de projeto", f"SMYS {inp['smys']:.0f} MPa / F = {inp['f_proj']:.2f}", "MAOP por Barlow"),
        ("MAOP (Barlow)", f"{res['p_adm_mpa']:.2f} MPa = {res['h_adm']:.0f} m polpa", ""),
        ("K de Durand", f"{inp['k_durand']:.0f}", "Só se d50 >= 40 um (não usado)"),
        ("Peso de declividade", f"{inp['peso']:.2f}", "Custo = declividade x peso + 1"),
        ("Rampa máxima do terreno", (f"{inp.get('rampa_max_pct', 0):.0f} % (penalidade {inp.get('custo_rampa', 0):.0e} por célula)"
                                     if inp.get('limitar_rampa') else "sem limite"), "Declividade da célula do MDE reamostrado"),
        ("Simplificação Douglas-Peucker", f"{inp['tol_geo']} graus", "Só o traçado exibido"),
        ("Faixa de APP (hidrografia)", f"{inp['app_m']} m", ""),
        ("Penalidades (desenho/hidro/gravidade/inválida)",
         f"{inp.get('peso_desenho', CUSTO_RESTRICAO_DESENHO_MANUAL):.0e} / {inp.get('peso_hidro', CUSTO_RESTRICAO_HIDROGRAFIA):.0e} / {CUSTO_PENALIDADE_GRAVIDADE:.0e} / {CUSTO_AREA_INVALIDA:.0e}", ""),
        ("Pesos por shapefile de restrição (custo/célula; buffer)",
         "; ".join(f"{k}: {v[0]:.0e}; {v[1]:.0f} m" for k, v in inp.get('pesos_camadas', {}).items()) or "nenhum", "Definidos na aba Pesos das Camadas"),
        ("MDE: arquivo / CRS", f"{inp['tiff_name']} / {res['crs']}", "Informar fonte do MDE"),
        ("MDE: pixel x fator de reamostragem", f"{inp['pixel_size']:.6g} (unid. CRS) x {inp['fator']}", "Resolução efetiva"),
        ("Origem (lat, lon / cota)", f"{res['lat_o']:.5f}, {res['lon_o']:.5f} / {res['cota_origem']:.1f} m", ""),
        ("Destino (lat, lon / cota)", f"{res['lat_d']:.5f}, {res['lon_d']:.5f} / {res['cota_destino']:.1f} m", ""),
        ("Distância geodésica origem-destino", f"{res['dist_reta_km']:.2f} km (sinuosidade {res['sinuosidade']:.3f})", "Extensão/distância reta"),
        ("Intervalo médio de amostragem do perfil", f"{res['passo_medio_m']:.0f} m", "Geodésica (WGS84) entre amostras"),
        ("Tolerância de violação", f"{TOL_VIOLACAO_M} m", "Trechos críticos"),
    ]


def gerar_imagens_temp(res_n, res_g, fator, tiff_path, sid, regra_slack=20):
    path_perfil = f"temp_perfil_{sid}.png"
    path_mapa = f"temp_mapa_{sid}.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    if res_n:
        ax1.plot(res_n['dist_arr'], res_n['elev'], color='red', alpha=0.3, label='Terreno (Otimizada)')
        ax1.plot(res_n['dist_arr'], res_n['hgl'], color='blue', label='HGL (Otimizada)')
        ax1.plot(res_n['dist_arr'], res_n['elev'] + regra_slack, color='orange', alpha=0.5, linestyle=':', label=f'Limite Slack Flow (+{regra_slack} m)')
        ax2.plot(res_n['dist_arr'], res_n['pressao_mpa'], color='blue', label='Pressão interna (MPa)')
        ax2.axhline(res_n['p_adm_mpa'], color='red', linestyle='--', label=f"MAOP = {res_n['p_adm_mpa']:.1f} MPa")
    if res_g:
        ax1.plot(res_g['dist_arr'], res_g['elev'], color='purple', alpha=0.3, linestyle='--', label='Terreno (Gravidade)')
        ax1.plot(res_g['dist_arr'], res_g['hgl'], color='lightblue', linestyle='--', label='HGL (Gravidade)')
    ax1.set_title("Perfil Hidráulico Comparativo")
    ax1.set_ylabel("Cota / HGL (m de polpa)")
    ax2.set_xlabel("Distância (km)")
    ax2.set_ylabel("Pressão (MPa)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_perfil, dpi=100)
    plt.close(fig)

    try:
        with rasterio.open(tiff_path) as src:
            new_shape = (max(1, int(src.height / fator)), max(1, int(src.width / fator)))
            data = src.read(1, out_shape=new_shape)
            plt.figure(figsize=(8, 8))
            plt.imshow(data, cmap='terrain', alpha=0.8)
            for res, estilo, nome in ((res_n, 'r-', 'Otimizada'), (res_g, 'm--', 'Gravidade')):
                if res:
                    xs, ys = transform('EPSG:4326', src.crs, res['lons_draw'], res['lats_draw'])
                    rc = [src.index(x, y) for x, y in zip(xs, ys)]
                    plt.plot(np.array([c[1] for c in rc]) / fator, np.array([c[0] for c in rc]) / fator,
                             estilo, linewidth=2, label=nome)
            plt.axis('off')
            plt.legend()
            plt.tight_layout()
            plt.savefig(path_mapa, dpi=100)
    except Exception as e:
        st.warning(f"Não foi possível gerar a imagem do mapa para o PDF: {e}")
    finally:
        plt.close('all')
    return path_perfil, path_mapa

def gerar_pdf_bytes(inputs, res_n, res_g, fator, tiff_path, sid):
    regra_slack = inputs['slack']
    path_perfil, path_mapa = gerar_imagens_temp(res_n, res_g, fator, tiff_path, sid, regra_slack)
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.chapter_title('1. DADOS DE ENTRADA DO PROJETO (REPRODUTIBILIDADE)')
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(60, 5, "Parametro", 1); pdf.cell(80, 5, "Valor adotado", 1); pdf.cell(50, 5, "Observacao", 1); pdf.ln()
    pdf.set_font('Arial', '', 7)
    for p, v, o in montar_parametros(inputs, res_n):
        pdf.cell(60, 5, pdf_safe(p)[:42], 1); pdf.cell(80, 5, pdf_safe(v)[:62], 1); pdf.cell(50, 5, pdf_safe(o)[:36], 1); pdf.ln()
    pdf.ln(4)

    pdf.chapter_title('2. MODELO FISICO E EQUACOES UTILIZADAS')
    pdf.set_font('Courier', '', 8)
    formulas = (
        "1. SG = 100 / [ Cw/SG_solido + (100-Cw)/1 ]\n"
        "2. Q [m3/s] = Producao[t/ano] / (Dias*24*Disp) / (Cw/100 * SG * 3600)\n"
        "3. V = Q / Area_interna ; D_interno = OD - 2*espessura\n"
        "4. f = 0.25 / [log10(k/(3.7D) + 5.74/Re^0.9)]^2 ; i_agua = f*V^2/(2*g*D)   [m.c.a/m]\n"
        "5. FL = 0.4794 + 0.5429*(0.01*Cv)^0.1058*(log10(d50) - 1), d50 em MICROMETROS\n"
        "   (ajuste empirico de terceiros; Durand-Condolios publicaram FL em grafico)\n"
        "   VL = FL * sqrt(2*g*D*(SG_solido - 1))\n"
        "6. d50 < 40 um (pseudo-homogeneo): J[m.c.a/m] = SG*i_agua  ->  J_polpa[m polpa/m] = i_agua\n"
        "   d50 >= 40 um: Phi = K*[V^2/(g*D*(SG_s-1))]^-1.5 ; J[m.c.a/m] = i_agua*(1+Cv*Phi)\n"
        "                 J_polpa = J[m.c.a/m] / SG   (valido so para V >= VL)\n"
        "7. HGL[i] = z_origem + H1 - J_polpa*Dist[i]  (+ H2 a jusante da EB-2)   [m de polpa]\n"
        "   H = P / (rho_m*g) ; Pressao[i] = rho_m*g*(HGL[i] - z[i])   [MPa]\n"
        "8. MAOP = 2*SMYS*t*F/OD (Barlow)\n"
        "9. Pot = rho_m*g*Q*H_total / (eta*1e6)   [MW]"
    )
    pdf.multi_cell(0, 4, formulas)
    pdf.ln(4)

    pdf.chapter_title('3. MEMORIAL DE CALCULOS HIDRAULICOS (ROTA OTIMIZADA)')
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(90, 6, "Variavel", 1); pdf.cell(100, 6, "Valor calculado", 1); pdf.ln()
    pdf.set_font('Arial', '', 9)
    risco = "SIM (referencia)" if res_n['v_op'] < res_n['v_limite_deposicao'] else "Nao (V >= VL)"
    regime = "Pseudo-homogeneo" if res_n['regime_hidraulico'] == 'homogeneo' else "Heterogeneo (Durand-Condolios)"
    cfg = "com EB-1 + EB-2" if inputs['eb2'] else "somente EB-1"
    calc = [
        ("Gravidade especifica (SG)", f"{res_n['sg']:.4f}"),
        ("Vazao volumetrica", f"{res_n['vazao_m3s']*3600:.2f} m3/h ({res_n['vazao_m3s']:.4f} m3/s)"),
        ("Concentracao volumetrica (Cv)", f"{res_n['cv_pct']:.2f} %"),
        ("Velocidade de operacao (V)", f"{res_n['v_op']:.3f} m/s"),
        ("Regime adotado", regime),
        ("Fator de Durand (FL)", f"{res_n['fl_durand']:.3f}"),
        ("Velocidade limite (VL)", f"{res_n['v_limite_deposicao']:.2f} m/s"),
        ("V < VL ?", risco),
        ("J (m de polpa/m)", f"{res_n['j_gradiente']:.6f}"),
        ("J (m.c.a/m)", f"{res_n['j_mca']:.6f}"),
        ("Extensao da rota", f"{res_n['dist_total']:.2f} km"),
        ("Perda total por atrito", f"{res_n['perda_total_m']:.0f} m de polpa"),
        ("Carga requerida (maximo do perfil; km)", f"{res_n['h_req']:.0f} m de polpa = {res_n['h_req']*res_n['rho_m']*9.81/1e6:.2f} MPa"),
        (f"Carga instalada ({cfg})", f"{res_n['h_inst']:.0f} m de polpa"),
        ("Residual no terminal", f"{res_n['residual_terminal']:.0f} m de polpa"),
        ("Pressao maxima / MAOP", f"{res_n['p_max_mpa']:.2f} / {res_n['p_adm_mpa']:.2f} MPa"),
        ("Km da pressao maxima", f"{res_n.get('km_pmax', float('nan')):.2f} km"),
        ("Extensao acima da MAOP", f"{res_n.get('ext_acima_maop_km', float('nan')):.2f} km"),
        ("Travessias da faixa de hidrografia", f"{res_n.get('travessias_hidro')} (extensao na faixa: {res_n.get('ext_em_app_km')} km)"),
        ("Declividade máx. do terreno na rota / cota máx.", f"{100*res_n.get('decl_max_rota', float('nan')):.1f} % / {res_n.get('cota_max_rota', float('nan')):.0f} m; "
                                                             f"células acima da rampa: {res_n.get('cel_acima_rampa', 0)} de {res_n.get('cel_rota', 0)}"),
        ("Re / f de Darcy", f"{res_n.get('Re', float('nan')):.3g} / {res_n.get('f_darcy', float('nan')):.5f}"),
        ("Numero minimo de estacoes (MAOP)", f"{res_n['n_est_min']}"),
        (f"Potencia de eixo - instalada ({cfg}), eta={inputs['eta']:.2f}", f"{res_n['potencia']:.2f} MW"),
        ("Potencia de eixo - carga minima requerida", f"{res_n['potencia_min']:.2f} MW"),
    ]
    for a, b in calc:
        pdf.cell(90, 6, pdf_safe(a)[:55], 1); pdf.cell(100, 6, pdf_safe(b)[:62], 1); pdf.ln()
    if res_n.get('aviso_modelo'):
        pdf.ln(2); pdf.set_font('Arial', 'I', 8)
        pdf.multi_cell(0, 4.5, pdf_safe(f"Nota sobre o modelo hidraulico: {res_n['aviso_modelo']}"))

    pdf.ln(4)
    pdf.chapter_title('4. ANALISE DE SLACK FLOW, PRESSAO E EB-2')
    margem = res_n['hgl'] - res_n['elev']
    trechos = trechos_criticos(res_n['dist_arr'], margem, regra_slack)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 4.5, pdf_safe(f"Intervalo medio de amostragem: {res_n['passo_medio_m']:.0f} m. Tolerancia de violacao: {TOL_VIOLACAO_M} m."))
    if trechos:
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 5, pdf_safe(f"ALERTA: trechos com folga inferior a {regra_slack} m (configuracao: {cfg})."))
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 8)
        for t_, w in (("Inicio (km)", 40), ("Fim (km)", 40), ("Extensao (km)", 40), ("Folga minima (m)", 70)):
            pdf.cell(w, 5, t_, 1)
        pdf.ln(); pdf.set_font('Arial', '', 8)
        for t in trechos:
            pdf.cell(40, 5, f"{t['ini']:.2f}", 1); pdf.cell(40, 5, f"{t['fim']:.2f}", 1)
            pdf.cell(40, 5, f"{t['ext']:.2f}", 1); pdf.cell(70, 5, f"{t['p_min']:.2f}", 1); pdf.ln()
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.multi_cell(0, 5, pdf_safe(f"SUCESSO: HGL pelo menos {regra_slack} m acima do terreno em todo o tracado ({cfg})."))
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2); pdf.set_font('Arial', '', 9)
    a = res_n['analise_eb2']
    if a['necessaria']:
        txt = (f"Somente com EB-1 a 1a violacao ocorre no km {a['km_primeira_violacao']:.2f}. ")
        if a['km_sugerido'] is not None:
            txt += (f"Posicao sugerida da EB-2 (ultima com folga >= {regra_slack}+{inputs['p_succ_min_m']} m): km {a['km_sugerido']:.2f}; "
                    f"carga minima da EB-2 nessa posicao: {a['h2_min_sugerido']:.0f} m de polpa "
                    f"({a['h2_min_sugerido']*res_n['rho_m']*9.81/1e6:.2f} MPa). ")
        else:
            txt += "Nao ha posicao a montante com folga de succao suficiente. "
        if a['h2_min_usuario'] is not None:
            txt += f"Carga minima para a EB-2 posicionada pelo usuario: {a['h2_min_usuario']:.0f} m de polpa. "
        pdf.multi_cell(0, 4.5, pdf_safe(txt))
    if res_n.get('p_succao_eb2_m') is not None:
        pdf.multi_cell(0, 4.5, pdf_safe(f"Pressao de succao na EB-2 (sem a carga dela): {res_n['p_succao_eb2_m']:.0f} m de polpa; "
                                        f"pressao de descarga: {res_n['p_descarga_eb2_mpa']:.2f} MPa (MAOP {res_n['p_adm_mpa']:.2f} MPa)."))

    pdf.add_page()
    pdf.chapter_title('5. ANEXOS GRAFICOS')
    if os.path.exists(path_perfil):
        pdf.cell(0, 8, "Figura 1: Perfil topografico, HGL e pressao", 0, 1, 'C')
        pdf.image(path_perfil, x=10, w=190)
    if os.path.exists(path_mapa):
        pdf.ln(5)
        pdf.cell(0, 8, "Figura 2: Tracado em planta", 0, 1, 'C')
        pdf.image(path_mapa, x=40, w=130)
    for pth in (path_perfil, path_mapa):
        try:
            if os.path.exists(pth): os.remove(pth)
        except OSError as e:
            st.warning(f"Não foi possível remover arquivo temporário: {e}")
    return pdf.output(dest='S').encode('latin-1')

# --- 3. FUNÇÕES AUXILIARES GIS ---
@st.cache_data
def get_tiff_bounds(tiff_path):
    try:
        with rasterio.open(tiff_path) as src: 
            # Converte o BoundingBox do Rasterio para uma tupla nativa do Python
            return tuple(src.bounds)
    except Exception as e:
        st.error(f"Erro ao ler os limites do TIFF: {e}")
        return None
@st.cache_data
def carregar_terreno(fator, tiff_path):
    try:
        with rasterio.open(tiff_path) as src:
            # Le o raster ja reamostrado na resolucao reduzida (decimated read do GDAL),
            # em vez de carregar o TIFF inteiro em memoria e so depois fatiar com [::fator].
            # Isso reduz muito o pico de RAM para TIFFs grandes (proximos de 1GB).
            out_height = max(1, math.ceil(src.height / fator))
            out_width = max(1, math.ceil(src.width / fator))
            data_red = src.read(
                1,
                out_shape=(out_height, out_width),
                resampling=Resampling.average
            )
            pixel_size = abs(src.transform.a)
            return data_red, src.transform, src.crs, src.bounds, pixel_size
    except Exception as e:
        st.error(f"Erro ao carregar o TIFF de topografia: {e}")
        return None, None, None, None, None

def pixel_latlon(src, r, c):
    if np.isscalar(r): r = [r]
    if np.isscalar(c): c = [c]
    xs, ys = rasterio.transform.xy(src.transform, r, c)
    lons, lats = transform(src.crs, 'EPSG:4326', xs, ys)
    return lats, lons

def latlon_utm(src, lat, lon):
    xs, ys = transform('EPSG:4326', src.crs, [lon], [lat])
    return xs[0], ys[0]

def rasterizar_shp_simples(gdf, crs, transform_base, shape, fator, custo_fixo):
    try:
        if gdf.crs != crs: gdf = gdf.to_crs(crs)
        shapes = ((g, 1) for g in gdf.geometry if not g.is_empty)
        t = transform_base
        trans_red = rasterio.Affine(t.a * fator, t.b, t.c, t.d, t.e * fator, t.f)
        mask = features.rasterize(shapes, out_shape=shape, transform=trans_red, fill=0, dtype='uint8')
        return mask * custo_fixo
    except Exception as e:
        st.sidebar.warning(f"Falha ao rasterizar restrição: {e}")
        return np.zeros(shape)

def corrigir_geometrias_invalidas(gdf):
    """
    Repara geometrias topologicamente invalidas (auto-intersecao, aneis mal
    fechados etc.) — comum em shapefiles publicos de Unidades de Conservacao,
    APPs e outras bases fundiarias/ambientais. Sem isso, gpd.clip/overlay
    (que dependem de operacoes topologicas do GEOS) derrubam a app com
    TopologyException assim que encontram uma geometria assim.
    """
    if gdf is None or gdf.empty:
        return gdf
    invalidas = ~gdf.geometry.is_valid
    if invalidas.any():
        gdf = gdf.copy()
        gdf.loc[invalidas, 'geometry'] = gdf.loc[invalidas, 'geometry'].apply(make_valid)

        def so_partes_poligonais(geom):
            # make_valid pode devolver uma GeometryCollection (mistura de
            # polygon/linha/ponto) quando a geometria original tinha defeitos
            # mais complexos; para uma camada de restricao/area, so as partes
            # poligonais interessam.
            if geom is None or geom.is_empty or geom.geom_type in ('Polygon', 'MultiPolygon'):
                return geom
            if geom.geom_type == 'GeometryCollection':
                polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                if not polys:
                    return None
                partes = []
                for g in polys:
                    partes.extend(g.geoms if g.geom_type == 'MultiPolygon' else [g])
                return partes[0] if len(partes) == 1 else MultiPolygon(partes)
            return geom  # linhas/pontos (ex: hidrografia) ficam como estao

        gdf['geometry'] = gdf['geometry'].apply(so_partes_poligonais)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf

os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")   # GDAL reconstroi o .shx se faltar

def _geo_para_mapa(g):
    """Remove geometrias vazias/None e explode GeometryCollection (o folium quebra com KeyError: 'coordinates')."""
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    colec = g.geom_type == 'GeometryCollection'
    if colec.any():
        g = pd.concat([g[~colec], g[colec].explode(index_parts=False)])
        g = g[g.geometry.notna() & ~g.geometry.is_empty & (g.geom_type != 'GeometryCollection')]
    return g

def _fonte_vetorial_zip(zpath):
    """Localiza o arquivo vetorial dentro do .zip (aceita subpastas), em vez de assumir que
    o .shp esta na raiz. Retorna o caminho 'zip://...!interno' e a lista de arquivos do zip."""
    with zipfile.ZipFile(zpath) as z:
        nomes = [n for n in z.namelist()
                 if not n.endswith('/') and '__MACOSX' not in n and not os.path.basename(n).startswith('._')]
    for ext in ('.shp', '.gpkg', '.geojson', '.json', '.kml'):
        cand = sorted([n for n in nomes if n.lower().endswith(ext)], key=lambda n: (n.count('/'), len(n)))
        if cand:
            # Extrai para disco (pasta gravavel): o GDAL nao consegue gerar o .shx ausente dentro de um zip (/vsizip é só leitura).
            destino = os.path.join(os.path.dirname(zpath), 'extraido')
            with zipfile.ZipFile(zpath) as z:
                z.extractall(destino)
            return os.path.join(destino, *cand[0].split('/')), nomes
    raise ValueError("o .zip não contém .shp/.gpkg/.geojson/.kml. Conteúdo: " + ", ".join(nomes[:12]) +
                     (" ..." if len(nomes) > 12 else ""))

@st.cache_data(show_spinner="Lendo shapefile...")
def ler_zip_otimizado(up, bbox_filter=None, bbox_crs=None):
    """
    Le um shapefile de dentro de um .zip, opcionalmente recortando por bbox.

    Duas correcoes importantes em relacao a versao original:
    1. CRS do bbox: 'bbox_filter' vem em coordenadas do raster (normalmente UTM,
       em metros), mas o shapefile pode estar em outra CRS (ex: EPSG:4326, graus).
       Passar o bbox direto pro gpd.read_file sem reprojetar faz o filtro nao
       bater com nada (ou bater errado) -- na pratica o GDAL acaba lendo o
       arquivo inteiro (ou vazio) em vez de so a janela de interesse, o que e
       uma das causas do carregamento lento. Aqui a gente detecta a CRS nativa
       do shapefile (lendo so 1 feicao, bem barato) e reprojeta o bbox para ela
       antes de filtrar.
    2. Cache: esta funcao roda toda vez que QUALQUER widget do app muda (e o
       Streamlit reexecuta o script inteiro do zero a cada interacao). Sem
       @st.cache_data, o zip era re-extraido e re-parseado do disco a cada
       clique no mapa, cada slider movido etc. -- mesmo que o upload nao tivesse
       mudado. O cache faz isso rodar so uma vez por arquivo/bbox.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "temp.zip"), "wb") as f: f.write(up.getbuffer())
            zip_path, _nomes_zip = _fonte_vetorial_zip(os.path.join(tmp, 'temp.zip'))

            bbox_tuple = None
            if bbox_filter and bbox_crs:
                bbox_filter = tuple(float(v) for v in bbox_filter)
                # Descobre a CRS nativa do shapefile lendo so 1 feicao (barato,
                # nao carrega o arquivo inteiro so pra checar a projecao).
                shp_crs = gpd.read_file(zip_path, rows=1).crs
                if shp_crs is not None and str(shp_crs) != str(bbox_crs):
                    bbox_tuple = transform_bounds(bbox_crs, shp_crs, *bbox_filter)
                else:
                    bbox_tuple = bbox_filter
                gdf = gpd.read_file(zip_path, bbox=bbox_tuple)
            else:
                gdf = gpd.read_file(zip_path)

            if gdf.empty: return None

            gdf = corrigir_geometrias_invalidas(gdf)
            if gdf is None or gdf.empty: return None

            if bbox_filter and bbox_crs:
                # Recorte fino: reprojeta o resultado ja pre-filtrado (poucas
                # feicoes) para a CRS do raster e clipa exatamente no bbox original.
                gdf = gdf.to_crs(bbox_crs)
                box_poly = box(*bbox_filter)
                gdf = gpd.clip(gdf, box_poly)
            return gdf
    except Exception as e:
        st.sidebar.warning(f"Falha ao ler shapefile '{up.name}': {e}")
        return None

@st.cache_data(show_spinner="Processando camada de restrição...")
def preparar_custo_shapefile(up, bbox_filter, bbox_crs, transform_gdal, shape, fator,
                              custo_fixo, buffer_m=0.0):
    """
    Le (via ler_zip_otimizado, ja cacheada) + opcionalmente bufferiza + rasteriza
    um shapefile em uma unica funcao cacheada.

    Por que isso importa especialmente para hidrografia: redes de drenagem
    nacionais (ANA etc.) tem MUITO mais feicoes e vertices por feicao do que um
    shapefile tipico de restricao (ex: 1 poligono de UC). Antes, o buffer() e o
    rasterize() rodavam FORA de qualquer cache, entao eram refeitos do zero a
    cada interacao no app (mover um slider, clicar no mapa) mesmo sem o
    shapefile ou o buffer terem mudado -- e buffer() em milhares de linhas
    densas e uma das operacoes mais caras do shapely. Envolvendo tudo numa
    unica funcao cacheada por (arquivo, bbox, buffer, resolucao), isso so roda
    de verdade quando algum desses parametros realmente muda.

    Retorna (matriz_de_custo, gdf_original_para_exibicao_no_mapa).
    """
    gdf = ler_zip_otimizado(up, bbox_filter=bbox_filter, bbox_crs=bbox_crs)
    if gdf is None:
        return None, None

    gdf_dst = gdf if str(gdf.crs) == str(bbox_crs) else gdf.to_crs(bbox_crs)

    if buffer_m and buffer_m > 0:
        gdf_dst = gdf_dst.copy()
        # buffer() so da resultado correto numa CRS projetada (unidades em metros).
        # Se a CRS de destino (a do proprio raster/TIFF) for geografica (graus),
        # buffer_m seria interpretado como GRAUS, nao metros -- um buffer gigante
        # e sem sentido fisico (e a origem do aviso "Results from 'buffer' are
        # likely incorrect" do geopandas). Nesse caso, faz o buffer numa UTM
        # estimada automaticamente pela extensao dos proprios dados, e so depois
        # reprojeta de volta pra CRS do raster antes de rasterizar.
        crs_original = gdf_dst.crs
        usar_utm_temp = crs_original is not None and crs_original.is_geographic
        if usar_utm_temp:
            crs_buffer = gdf_dst.estimate_utm_crs()
            gdf_dst = gdf_dst.to_crs(crs_buffer)

        # Simplifica antes de bufferizar: reduz drasticamente o numero de
        # vertices sem alterar a forma de forma perceptivel na escala do
        # buffer aplicado. resolution=4 (em vez do default 8) tambem corta
        # pela metade os pontos gerados em cada curva do buffer.
        gdf_dst['geometry'] = gdf_dst.geometry.simplify(max(buffer_m / 4, 1.0), preserve_topology=True)
        gdf_dst['geometry'] = gdf_dst.geometry.buffer(buffer_m, resolution=4)

        if usar_utm_temp:
            gdf_dst = gdf_dst.to_crs(crs_original)

    transform_base = rasterio.Affine.from_gdal(*transform_gdal)
    custo = rasterizar_shp_simples(gdf_dst, bbox_crs, transform_base, shape, fator, custo_fixo)
    return custo, gdf

# --- 4. INTERFACE E INPUTS ---
if 'inicio_coords' not in st.session_state: 
    st.session_state.update({'inicio_coords': None, 'fim_coords': None, 'eb2_coords': None, 'user_drawings': []})
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = uuid.uuid4().hex[:8]
sid = st.session_state['session_id']

st.sidebar.title("🎛️ Painel de Controle")

st.sidebar.header("📁 1. Mapa Base (Topografia)")
def baixar_dtm_earthengine(asset_id, projeto, lon_min, lat_min, lon_max, lat_max, escala_m, saida):
    """Baixa um DTM do Google Earth Engine (ex.: ANADEM, FABDEM) ja em EPSG:5880 (SIRGAS 2000 / Brazil Polyconic),
    em blocos (limite de download direto do EE), e grava um GeoTIFF unico. Requer: pip install earthengine-api
    e autenticacao previa (terminal: earthengine authenticate)."""
    import ee, io
    from rasterio.transform import from_origin
    try:
        ee.Initialize(project=projeto)
    except Exception:
        ee.Initialize()
    xs, ys = transform('EPSG:4326', 'EPSG:5880', [lon_min, lon_max, lon_min, lon_max], [lat_min, lat_min, lat_max, lat_max])
    x0 = math.floor((min(xs) - escala_m) / escala_m) * escala_m
    x1 = math.ceil((max(xs) + escala_m) / escala_m) * escala_m
    y1 = math.ceil((max(ys) + escala_m) / escala_m) * escala_m
    y0 = math.floor((min(ys) - escala_m) / escala_m) * escala_m
    W, H = int((x1 - x0) / escala_m), int((y1 - y0) / escala_m)
    tipo = ee.data.getAsset(asset_id)['type']
    img = ee.ImageCollection(asset_id).mosaic() if tipo == 'IMAGE_COLLECTION' else ee.Image(asset_id)
    img = img.select(0).toFloat()
    if tipo == 'IMAGE_COLLECTION':
        proj_nativa = ee.Image(ee.ImageCollection(asset_id).first()).select(0).projection()
        img = img.setDefaultProjection(proj_nativa)
    proj_alvo = ee.Projection('EPSG:5880').atScale(escala_m)
    img = img.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(proj_alvo)
    mosaico = np.full((H, W), np.nan, dtype='float32')
    B = 1200   # pixels por bloco (cabe no limite de ~32 MB por download)
    blocos = [(r, c) for r in range(0, H, B) for c in range(0, W, B)]
    prog = st.progress(0.0, text="Baixando blocos do Earth Engine...")
    import requests
    for k, (r, c) in enumerate(blocos):
        h, w = min(B, H - r), min(B, W - c)
        url = img.getDownloadURL({'name': 'dtm', 'format': 'GEO_TIFF', 'crs': 'EPSG:5880',
                                  'crsTransform': [escala_m, 0, x0 + c * escala_m, 0, -escala_m, y1 - r * escala_m],
                                  'dimensions': f'{w}x{h}'})
        resp = requests.get(url, timeout=300); resp.raise_for_status()
        with rasterio.open(io.BytesIO(resp.content)) as ds:
            mosaico[r:r + h, c:c + w] = ds.read(1)[:h, :w]
        prog.progress((k + 1) / len(blocos), text=f"Bloco {k + 1}/{len(blocos)}")
    mosaico = np.nan_to_num(mosaico, nan=0.0)   # sem dado (ex.: mar) = cota 0
    with rasterio.open(saida, 'w', driver='GTiff', height=H, width=W, count=1, dtype='float32',
                       crs='EPSG:5880', transform=from_origin(x0, y1, escala_m, escala_m), compress='deflate') as dst:
        dst.write(mosaico, 1)
    return saida

fonte_mde = st.sidebar.radio("Fonte do MDE", ("Upload TIFF", "Baixar DTM (Google Earth Engine)"), index=0)
tiff_path = None
if fonte_mde == "Upload TIFF":
    tiff_file = st.sidebar.file_uploader("Upload TIFF", type=["tif", "tiff"])
    if tiff_file:
        tiff_path = f"temp_topografia_{sid}.tif"
        with open(tiff_path, "wb") as f: f.write(tiff_file.getbuffer())
    elif os.path.exists("topografia_utm.tif"):
        tiff_path = "topografia_utm.tif"
    else:
        st.warning("Faça o upload do TIFF para iniciar.")
        st.stop()
else:
    st.sidebar.caption("Requer `pip install earthengine-api` e `earthengine authenticate` (uma vez, no terminal). "
                       "O DTM é recortado e reprojetado automaticamente para EPSG:5880.")
    proj_ee = st.sidebar.text_input("ID do projeto Google Cloud/Earth Engine", value="")
    modelo_dtm = st.sidebar.selectbox("Modelo", ("ANADEM (ANA, 30 m)", "FABDEM (30 m, uso não comercial)", "Outro asset"))
    assets_padrao = {"ANADEM (ANA, 30 m)": "projects/et-brasil/assets/anadem/v1",
                     "FABDEM (30 m, uso não comercial)": "projects/sat-io/open-datasets/FABDEM"}
    asset_id = st.sidebar.text_input("Asset do Earth Engine", value=assets_padrao.get(modelo_dtm, ""),
                                     help="Confira o ID atual no catálogo do Earth Engine / awesome-gee-community-catalog.")
    cb1, cb2 = st.sidebar.columns(2)
    lon_min = cb1.number_input("Lon mín", value=-43.7, format="%.4f"); lon_max = cb2.number_input("Lon máx", value=-40.8, format="%.4f")
    lat_min = cb1.number_input("Lat mín", value=-21.9, format="%.4f"); lat_max = cb2.number_input("Lat máx", value=-18.8, format="%.4f")
    escala_m = st.sidebar.selectbox("Resolução do recorte (m)", (90, 60, 30), index=0,
                                    help="30 m gera ~9x mais células que 90 m; use também o controle de resolução do cálculo.")
    chave = f"{asset_id}|{lon_min}|{lat_min}|{lon_max}|{lat_max}|{escala_m}"
    saida = os.path.join(tempfile.gettempdir(), f"dtm_{__import__("hashlib").md5(chave.encode()).hexdigest()[:12]}.tif")
    if st.sidebar.button("⬇️ Baixar/atualizar DTM"):
        try:
            with st.spinner("Baixando DTM..."):
                baixar_dtm_earthengine(asset_id, proj_ee or None, lon_min, lat_min, lon_max, lat_max, escala_m, saida)
            st.session_state['dtm_path'] = saida
            st.session_state['dtm_chave'] = chave
        except Exception as e:
            st.sidebar.error(f"Falha no download: {e}")
    if st.session_state.get('dtm_chave') == chave and os.path.exists(st.session_state.get('dtm_path', '')):
        tiff_path = st.session_state['dtm_path']
        st.sidebar.success("DTM carregado (EPSG:5880).")
    elif os.path.exists(saida):
        tiff_path = saida
    else:
        st.info("Preencha o projeto/asset e a área e clique em **Baixar/atualizar DTM**.")
        st.stop()

fator = st.sidebar.slider("Resolução do Cálculo (1=Alta, 20=Baixa)", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("🚧 2. Áreas de Restrição (Zips)")
up_shapes = st.sidebar.file_uploader("Upload de Shapefiles (.zip)", type="zip", accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("🌊 2b. Hidrografia (Rios e Corpos d'Água)")
up_hidrografia = st.sidebar.file_uploader(
    "Upload Shapefile de Rios (.zip)", type="zip", key="up_hidro",
    help="Linhas/polígonos de drenagem (ex: ANA, IBGE, OSM). Evita que a rota siga o curso do rio."
)
buffer_hidrografia = st.sidebar.number_input(
    "Faixa de Exclusão / APP (m)", 0, 1000, 30,
    help="Largura da faixa marginal de proteção aplicada a cada lado do rio (Código Florestal define minimos conforme a largura do curso d'água)."
)

st.sidebar.markdown("---")
st.sidebar.header("📐 3. Geometria da Rota")
peso_declive = st.sidebar.slider("Peso Declividade", 0.0, 20.0, 5.0)
limitar_rampa = st.sidebar.checkbox("Limitar rampa do terreno", value=True,
                                    help="Penaliza as células cuja declividade do terreno (na resolução do MDE reamostrado) "
                                         "excede a rampa máxima. O Minas-Rio foi construído com rampas de até 15%.")
rampa_max_pct = st.sidebar.number_input("Rampa máxima (%)", 1.0, 100.0, 15.0, step=1.0, disabled=not limitar_rampa)
custo_rampa = st.sidebar.number_input("Penalidade por célula acima da rampa", 0.0, 1e8, 1e4, step=1000.0, format="%.0f",
                                      disabled=not limitar_rampa,
                                      help="Custo somado a cada célula acima da rampa máxima. Valores altos tornam a célula "
                                           "quase intransponível; a rota só a cruza se não houver desvio mais barato.")
tol_geo = st.sidebar.slider("Simplificação (Graus)", 0.00001, 0.01, 0.0001, format="%.5f")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 4. Parâmetros Hidráulicos")
prod = st.sidebar.number_input("Produção (Mtpa)", 1.0, 100.0, 26.0)
conc = st.sidebar.slider("Conc. Sólidos (%)", 10, 80, 70)
dias_op = st.sidebar.number_input("Dias de operação por ano", 300, 366, 365)
disp = st.sidebar.number_input("Disponibilidade física", 0.50, 1.00, 0.92, step=0.01)
OD_OPCOES = [8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 36, 40]
od_pol = st.sidebar.selectbox("Diâmetro EXTERNO (pol) - API 5L", OD_OPCOES, index=OD_OPCOES.index(26),
                              help="Na API 5L o diâmetro nominal é o EXTERNO. O interno é calculado: OD - 2 x espessura.")
esp_mm = st.sidebar.number_input("Espessura de parede (mm)", 5.0, 60.0, 19.05, step=0.01)
diam_int_m = od_pol * 0.0254 - 2 * esp_mm / 1000
if diam_int_m <= 0:
    st.sidebar.error("Espessura incompatível com o diâmetro externo."); st.stop()
st.sidebar.caption(f"Diâmetro interno resultante: {diam_int_m*1000:.1f} mm")
sg_solido = st.sidebar.number_input("SG do Sólido (t/m³)", 1.5, 8.0, 4.6, help="Densidade real do minério (ex: 4.6 para hematita/itabirito)")
d50 = st.sidebar.number_input("D50 do Sólido (µm)", 10, 3000, 25, help="Diâmetro mediano em MICROMETROS (a Eq. de FL usa µm)")
rugosidade_mm = st.sidebar.number_input("Rugosidade Interna do Tubo (mm)", 0.001, 1.0, 0.045, format="%.3f", help="Aço ≈0.045mm | Polietileno ≈0.015mm")
visc_cp = st.sidebar.number_input("Viscosidade (cP)", 0.1, 50.0, 1.0, step=0.1)
visc_polpa = st.sidebar.checkbox("Viscosidade informada é a APARENTE da polpa (Re usa ρ da polpa)", value=False,
                                 help="Desmarcado: viscosidade do carreador (água, 1 cP). Marcado: viscosidade aparente da polpa (ex.: 5, 10, 20 cP) para análise de sensibilidade.")
k_durand = st.sidebar.slider("Constante de Durand (K)", 100.0, 150.0, 121.0, help="Forma SEM o termo √C_D (K=121). Usada só se d50 ≥ 40 µm. A forma com √C_D usa K=81.")
eta = st.sidebar.number_input("Rendimento global motor+bomba", 0.30, 1.00, 0.90, step=0.01)
regra_slack_flow = st.sidebar.number_input("Regra de Slack Flow (m)", 0, 100, 20, help="Folga mínima HGL - terreno (m de polpa)")
p_succ_min_m = st.sidebar.number_input("Pressão mínima de sucção da EB-2 (m de polpa)", 0, 200, 20, help="Usada no critério de posição da EB-2")
p_res_m = st.sidebar.number_input("Pressão residual mínima no terminal (m de polpa)", 0, 200, 20)
st.sidebar.markdown("**Bombeamento (em pressão, MPa)**")
p1_mpa = st.sidebar.number_input("EB-1: pressão de descarga (MPa)", 0.0, 40.0, 17.66, step=0.1,
                                 help="Ref. Minas-Rio ≈ 1.835 m.c.a ≈ 18 MPa (1 MPa ≈ 101,9 m.c.a)")
modo_eb2 = st.sidebar.radio("EB-2", ("Sem EB-2", "Manual (clique/coordenadas)", "Automática (onde a folga é violada)"), index=0,
                            help="Automática: o SIAD calcula a rota só com a EB-1, acha a 1ª violação da folga e posiciona a EB-2 "
                                 "na última amostra a montante com folga ≥ regra + sucção mínima.")
usar_eb2 = modo_eb2 != "Sem EB-2"
eb2_auto = modo_eb2.startswith("Automática")
eb2_manual = usar_eb2 and not eb2_auto
p2_mpa = 0.0
h2_auto_dim = False
if usar_eb2:
    if eb2_auto:
        h2_auto_dim = st.sidebar.checkbox("Dimensionar incremento da EB-2 automaticamente (carga mínima)", value=True)
    if not h2_auto_dim:
        p2_mpa = st.sidebar.number_input("EB-2: incremento de pressão (MPa)", 0.0, 40.0, 19.62, step=0.1,
                                         help="Pressão de descarga = pressão de sucção + incremento. Ref. Minas-Rio ≈ 2.039 m.c.a ≈ 20 MPa")
st.sidebar.markdown("**Tubo (MAOP por Barlow)**")
smys_mpa = st.sidebar.number_input("SMYS do aço (MPa)", 200.0, 800.0, 485.0, help="API 5L X70 = 485 MPa")
f_proj = st.sidebar.number_input("Fator de projeto", 0.3, 1.0, 0.72, step=0.01)

terreno, trans, crs, bounds, pixel_size = carregar_terreno(fator, tiff_path)
if terreno is None: st.error("Erro TIFF."); st.stop()

_assinatura_mde = (str(crs), tuple(round(b, 6) for b in bounds), fator)
if st.session_state.get('assinatura_mde') != _assinatura_mde:
    # MDE (CRS/extensao) mudou: coordenadas guardadas da sessao anterior deixam de valer.
    st.session_state.update({'assinatura_mde': _assinatura_mde, 'inicio_coords': None, 'fim_coords': None, 'eb2_coords': None})
if st.session_state['inicio_coords'] is None:
    st.session_state['inicio_coords'] = (bounds.left*0.9+bounds.right*0.1, bounds.bottom*0.1+bounds.top*0.9)
    st.session_state['fim_coords'] = (bounds.left*0.1+bounds.right*0.9, bounds.bottom*0.9+bounds.top*0.1)

st.sidebar.markdown("---")
st.sidebar.header("📍 5. Posicionamento (Coordenadas)")
modo = st.sidebar.radio("Mover Clique no Mapa:", ("Nenhum", "🟢 Origem", "🔵 Destino", "📍 EB-2"), index=0)

c1, c2 = st.sidebar.columns(2)
mx = c1.number_input("Origem X", value=st.session_state['inicio_coords'][0], format="%.5f")
my = c2.number_input("Origem Y", value=st.session_state['inicio_coords'][1], format="%.5f")
st.session_state['inicio_coords'] = (mx, my)

c3, c4 = st.sidebar.columns(2)
px = c3.number_input("Destino X", value=st.session_state['fim_coords'][0], format="%.5f")
py = c4.number_input("Destino Y", value=st.session_state['fim_coords'][1], format="%.5f")
st.session_state['fim_coords'] = (px, py)

def _para_crs_raster(x, y):
    """Se o MDE esta em CRS projetado (metros) e o usuario digitou lon/lat em graus, converte."""
    if crs is not None and crs.is_projected and abs(x) <= 180 and abs(y) <= 90:
        xs_, ys_ = transform('EPSG:4326', crs, [x], [y])
        return float(xs_[0]), float(ys_[0])
    return x, y
st.session_state['inicio_coords'] = _para_crs_raster(mx, my)
st.session_state['fim_coords'] = _para_crs_raster(px, py)

if eb2_manual:
    if st.session_state['eb2_coords'] is None: st.session_state['eb2_coords'] = ((mx+px)/2, (my+py)/2)
    c5, c6 = st.sidebar.columns(2)
    ex = c5.number_input("EB-2 X", value=st.session_state['eb2_coords'][0], format="%.5f")
    ey = c6.number_input("EB-2 Y", value=st.session_state['eb2_coords'][1], format="%.5f")
    st.session_state['eb2_coords'] = _para_crs_raster(ex, ey)

# --- 5. MOTOR DE CUSTOS E DIJKSTRA ---
tiff_bounds = get_tiff_bounds(tiff_path)

# Custo por DECLIVIDADE REAL (gradiente entre pixels vizinhos), nao por cota bruta.
# Usar a elevacao absoluta (terreno * peso) fazia o vale do rio - a regiao de menor
# cota da bacia - virar sistematicamente o caminho "mais barato", entao a rota
# tendia a seguir o curso d'agua. Com o gradiente, o custo reflete o quanto o
# terreno varia ali, nao a altitude em si.
t_ini_calc = time.perf_counter()
_tempos = []   # (etapa, segundos, n_feicoes)
def calcular_declividade_m_m(terr, crs_, bounds_, pixel_size_, fator_):
    """Declividade (tangente, m/m). Se o MDE estiver em graus (CRS geografico), converte o
    espacamento da malha para metros por linha de latitude. Se estiver em CRS projetado
    (ex.: EPSG:5880), o espacamento ja e' em metros."""
    t = terr.astype(float)
    res = pixel_size_ * fator_
    if crs_ is not None and crs_.is_geographic:
        h = t.shape[0]
        lat_rows = bounds_.top - (np.arange(h) + 0.5) * res
        dy = res * 110574.0
        dx = res * 111320.0 * np.cos(np.radians(lat_rows))[:, None]
        gy_ = np.gradient(t, axis=0) / dy
        gx_ = np.gradient(t, axis=1) / dx
    else:
        gy_, gx_ = np.gradient(t, res)
    return np.hypot(gx_, gy_)
declividade = calcular_declividade_m_m(terreno, crs, bounds, pixel_size, fator)  # m/m
if crs is not None and crs.is_geographic:
    st.sidebar.warning("MDE em graus (geográfico): declividade convertida para m/m com espaçamento aproximado e "
                       "custo geométrico isotrópico em índices. Recomenda-se reprojetar o MDE (ex.: EPSG:5880, SIRGAS 2000 / Brazil Polyconic).")
st.title("Sistema de Otimização e Viabilidade de Minerodutos")
tab1, tab2, tab3, tab_pesos = st.tabs(["🌍 Mapa Interativo", "⚡ Perfil Hidráulico (HGL)", "📄 Relatório e Dados (Download)", "⚖️ Pesos das Camadas"])

PRESETS_PESO = {
    "Proibitivo (5×10⁶)": 5_000_000.0,
    "Muito alto – restrição rígida (5×10⁵)": 500_000.0,
    "Alto (5×10⁴)": 50_000.0,
    "Moderado (5×10³)": 5_000.0,
    "Baixo (5×10²)": 500.0,
    "Leve (50)": 50.0,
    "Personalizado": None,
}
def _peso_widget(rotulo, chave, padrao_preset):
    """Seletor de peso (custo adicional por célula) + valor personalizado. Retorna o custo por célula."""
    opcoes = list(PRESETS_PESO)
    c1, c2 = st.columns([3, 2])
    escolha = c1.selectbox(f"Peso – {rotulo}", opcoes, index=opcoes.index(padrao_preset), key=f"preset_{chave}")
    if PRESETS_PESO[escolha] is None:
        return float(c2.number_input("Custo por célula", 0.0, 1e8, 1000.0, step=100.0, key=f"custom_{chave}"))
    c2.caption(f"Custo por célula: {PRESETS_PESO[escolha]:,.0f}")
    return PRESETS_PESO[escolha]

custo_base = (declividade * peso_declive) + 1.0
_masc_rampa = (declividade > rampa_max_pct / 100.0) if limitar_rampa else np.zeros(declividade.shape, dtype=bool)
if limitar_rampa:
    custo_base[_masc_rampa] += custo_rampa
    st.sidebar.caption(f"Células acima de {rampa_max_pct:.0f}%: {100.0 * float(_masc_rampa.mean()):.1f}% do MDE.")
custo_base[terreno <= 0] += CUSTO_AREA_INVALIDA
_custo_celula_livre = float(np.nanmean(custo_base[terreno > 0])) if (terreno > 0).any() else 1.0

pesos_camadas = {}      # nome do zip -> (custo por celula, buffer m)
with tab_pesos:
    st.markdown("### Peso de cada camada")
    st.caption("O peso é um **custo adicional por célula** dentro da camada, somado a `declividade × peso + 1`. "
               "O caminho de custo mínimo só atravessa a camada se o desvio custar mais que a penalidade. "
               f"Custo médio de uma célula livre neste MDE ≈ {_custo_celula_livre:.1f}; portanto uma penalidade de "
               f"5×10⁵ equivale a um desvio de ~{5e5/_custo_celula_livre:,.0f} células. "
               "Valores baixos tornam a camada uma preferência (o traçado pode cruzá-la se compensar); valores muito altos a tornam quase intransponível.")
    st.markdown("**Shapefiles de restrição enviados**")
    if not up_shapes:
        st.info("Nenhum shapefile de restrição enviado. Envie os .zip na barra lateral (seção 2) e eles aparecerão aqui.")
    for _up in (up_shapes or []):
        with st.container(border=True):
            st.markdown(f"**{_up.name}**")
            _w = _peso_widget(_up.name, f"shp_{_up.name}", "Muito alto – restrição rígida (5×10⁵)")
            _b = st.number_input("Faixa de proteção (buffer, m) ao redor das feições", 0.0, 20000.0, 0.0, step=50.0, key=f"buf_{_up.name}")
            pesos_camadas[_up.name] = (_w, _b)
    st.markdown("**Hidrografia / APP**")
    CUSTO_RESTRICAO_HIDROGRAFIA = _peso_widget("hidrografia", "hidro", "Muito alto – restrição rígida (5×10⁵)")
    st.markdown("**Áreas desenhadas no mapa**")
    CUSTO_RESTRICAO_DESENHO_MANUAL = _peso_widget("áreas desenhadas", "desenho", "Proibitivo (5×10⁶)")
    if pesos_camadas:
        st.dataframe(pd.DataFrame([{"Camada": k, "Custo por célula": v[0], "Buffer (m)": v[1]} for k, v in pesos_camadas.items()]),
                     hide_index=True, use_container_width=True)

gdfs_carregados = []
gdf_hidro_carregado = None

with rasterio.open(tiff_path) as src:
    crs_str = str(src.crs)
    transform_gdal = src.transform.to_gdal()

    if up_shapes:
        for up in up_shapes:
            _w_shp, _b_shp = pesos_camadas.get(up.name, (CUSTO_RESTRICAO_SHAPEFILE, 0.0))
            _t0 = time.perf_counter()
            custo_shp, gdf_restricao = preparar_custo_shapefile(
                up, tiff_bounds, crs_str, transform_gdal, terreno.shape, fator,
                _w_shp, buffer_m=_b_shp
            )
            _tempos.append((f"Shapefile: {up.name}", time.perf_counter() - _t0, None if gdf_restricao is None else len(gdf_restricao)))
            if gdf_restricao is not None:
                gdfs_carregados.append(gdf_restricao)
                custo_base += custo_shp

    if up_hidrografia:
        # buffer + simplificacao + rasterizacao acontecem dentro da funcao
        # cacheada -- so reprocessa de fato se o arquivo, o bbox ou a faixa
        # (buffer_hidrografia) mudarem, nao a cada interacao no app.
        _t0 = time.perf_counter()
        custo_hidro, gdf_hidro_carregado = preparar_custo_shapefile(
            up_hidrografia, tiff_bounds, crs_str, transform_gdal, terreno.shape, fator,
            CUSTO_RESTRICAO_HIDROGRAFIA, buffer_m=buffer_hidrografia
        )
        _tempos.append((f"Hidrografia: {up_hidrografia.name}", time.perf_counter() - _t0,
                        None if gdf_hidro_carregado is None else len(gdf_hidro_carregado)))
        if custo_hidro is not None:
            custo_base += custo_hidro

    if st.session_state.get('user_drawings'):
        val = [f for f in st.session_state['user_drawings'] if f.get('geometry')]
        if val:
            fc = {"type": "FeatureCollection", "features": val}
            gdf_draw = gpd.GeoDataFrame.from_features(fc).set_crs(epsg=4326).to_crs(src.crs)
            gdf_draw = corrigir_geometrias_invalidas(gdf_draw)
            _t0 = time.perf_counter()
            custo_base += rasterizar_shp_simples(gdf_draw, src.crs, src.transform, terreno.shape, fator, CUSTO_RESTRICAO_DESENHO_MANUAL)
            _tempos.append(("Áreas desenhadas no mapa", time.perf_counter() - _t0, len(gdf_draw)))

pontos_idx = []
with rasterio.open(tiff_path) as src:
    def _checar_dentro(r_, c_, nome_):
        if not (0 <= r_ < src.height and 0 <= c_ < src.width):
            raise ValueError(f"{nome_} fora do MDE (linha {r_}, coluna {c_}). Confira as coordenadas: o MDE está em "
                             f"{src.crs} (limites X {src.bounds.left:.5g} a {src.bounds.right:.5g}, "
                             f"Y {src.bounds.bottom:.5g} a {src.bounds.top:.5g}). Pode digitar lon/lat em graus.")
    try:
        r, c = src.index(*st.session_state['inicio_coords']); _checar_dentro(r, c, 'Origem')
        pontos_idx.append((min(int(r/fator), terreno.shape[0]-1), min(int(c/fator), terreno.shape[1]-1)))
        if eb2_manual and st.session_state['eb2_coords']:
            r, c = src.index(*st.session_state['eb2_coords']); _checar_dentro(r, c, 'EB-2')
            pontos_idx.append((min(int(r/fator), terreno.shape[0]-1), min(int(c/fator), terreno.shape[1]-1)))
        r, c = src.index(*st.session_state['fim_coords']); _checar_dentro(r, c, 'Destino')
        pontos_idx.append((min(int(r/fator), terreno.shape[0]-1), min(int(c/fator), terreno.shape[1]-1)))
    except Exception as e:
        st.error(f"Erro nas coordenadas: {e}"); st.stop()

custo_grav = custo_base.copy()
cota_inicio = terreno[pontos_idx[0]]
custo_grav[terreno > cota_inicio] += CUSTO_PENALIDADE_GRAVIDADE

@st.cache_data(show_spinner=False)
def calcular_segmentos(pontos, matriz):
    try:
        segmentos = []
        split_idx = None
        for i in range(len(pontos)-1):
            idx, _ = route_through_array(matriz, pontos[i], pontos[i+1], fully_connected=True, geometric=True)
            if i == 0 and len(pontos) == 3: split_idx = len(idx) - 1
            if i > 0: idx = idx[1:] 
            segmentos.append(np.array(idx))
        return np.vstack(segmentos), split_idx
    except Exception as e:
        st.error(f"Erro ao calcular a rota: {e}")
        return None, None

with st.spinner("Calculando Rotas (Otimizada e Gravidade)..."):
    _t0 = time.perf_counter()
    rota_n, split_n = calcular_segmentos(pontos_idx, custo_base)
    _tempos.append(("Rota otimizada (Dijkstra)", time.perf_counter() - _t0, None))
    _t0 = time.perf_counter()
    rota_g, split_g = calcular_segmentos(pontos_idx, custo_grav)
    _tempos.append(("Rota por gravidade (Dijkstra)", time.perf_counter() - _t0, None))
    st.session_state['tempo_calc_s'] = time.perf_counter() - t_ini_calc
    st.session_state['tempos_etapas'] = list(_tempos)
    st.session_state['n_celulas_grafo'] = int(terreno.shape[0] * terreno.shape[1])

# --- 6. CÁLCULO HIDRÁULICO (HGL E FÍSICA) ---
D50_LIMITE_HOMOGENEO_MICRON = 40.0  # abaixo disso: regime pseudo-homogeneo (ver docstring)

def calcular_gradiente_durand(vel, diam_m, sg_mix, sg_solido_local, d50_micron, rugosidade_m,
                               k_durand_local=121.0, sg_liquido=1.0, visc_cp=1.0, visc_polpa=False):
    """
    Gradiente hidraulico da polpa em DUAS bases:
      J_mca  [m.c.a/m]      -> base de pressao (metros de coluna d'agua)
      J      [m de polpa/m] -> base compativel com cota do terreno (usada na HGL):  J = J_mca / SG

    - Pseudo-homogeneo (d50 < 40 um): J_mca = SG * i_agua (Wasp et al., 1977)  =>  J = i_agua.
    - Heterogeneo (d50 >= 40 um): Durand-Condolios, J_mca = i_agua*(1 + Cv*Phi), Phi = K*Fr^-1.5
      (forma sem sqrt(C_D); valida so para V >= VL).
    FL (Eq. 6) e' um ajuste empirico de terceiros com d50 em MICROMETROS (Durand e Condolios
    publicaram FL em forma grafica).
    """
    g = G
    vel = max(vel, 1e-6)
    cv_pct = np.clip((sg_mix - sg_liquido) / (sg_solido_local - sg_liquido) * 100, 0.1, 45)
    cv_frac = cv_pct / 100

    # visc_polpa=False: visc_cp e' a viscosidade do carreador (agua) -> nu = mu / rho_agua.
    # visc_polpa=True : visc_cp e' a viscosidade APARENTE da polpa -> nu = mu / rho_polpa
    # (Re = rho_m * V * D / mu). Sem isso, o Re ficaria subestimado por um fator ~SG.
    rho_visc = (sg_mix if visc_polpa else sg_liquido) * 1000
    visc_m2s = (visc_cp * 1e-3) / rho_visc
    re = max(vel * diam_m / visc_m2s, 4000)
    f_darcy = 0.25 / (np.log10((rugosidade_m / (3.7 * diam_m)) + (5.74 / re**0.9)))**2
    i_agua = f_darcy * (vel**2) / (2 * g * diam_m)

    fl = 0.4794 + 0.5429 * (0.01 * cv_pct)**0.1058 * (np.log10(d50_micron) - 1)
    fl = max(fl, 0.3)
    vl = fl * np.sqrt(2 * g * diam_m * (sg_solido_local - sg_liquido) / sg_liquido)

    modelo_valido, aviso, phi = True, None, None
    if d50_micron < D50_LIMITE_HOMOGENEO_MICRON:
        regime = 'homogeneo'
        j_mca = i_agua * (sg_mix / sg_liquido)
        aviso = (f"d50={d50_micron:.0f}um < {D50_LIMITE_HOMOGENEO_MICRON:.0f}um: escoamento pseudo-homogeneo (aproximacao). "
                 f"VL/FL de Durand sao apenas referencia. Para dimensionamento final, usar modelo reologico "
                 f"(Bingham: Wilson-Thomas) e ensaios; nao ha tensao de escoamento no modelo.")
    else:
        regime = 'heterogeneo'
        fr2 = max((vel**2) / (g * diam_m * (sg_solido_local / sg_liquido - 1)), 2 * fl**2)
        phi = k_durand_local * fr2**(-1.5)
        j_mca = i_agua * (1 + cv_frac * phi)
        if vel < vl:
            modelo_valido = False
            aviso = (f"V ({vel:.2f} m/s) abaixo de VL ({vl:.2f} m/s): Durand-Condolios extrapolada fora do dominio "
                     f"(valida so para V >= VL). J NAO E CONFIAVEL.")
    j_polpa = j_mca / (sg_mix / sg_liquido)
    return {'Re': re, 'f': f_darcy, 'J': j_polpa, 'J_mca': j_mca, 'FL': fl, 'VL': vl, 'Cv_pct': cv_pct, 'i_agua': i_agua, 'Phi': phi,
            'regime': regime, 'modelo_valido': modelo_valido, 'aviso': aviso}


@st.cache_data(show_spinner=False)
def processar_completo(rota_idx, tolerancia, split_point, tiff_path, fator, terreno, hid):
    if rota_idx is None: return None
    pr, pc = rota_idx[:, 0], rota_idx[:, 1]
    with rasterio.open(tiff_path) as src:
        lats, lons = pixel_latlon(src, pr*fator, pc*fator)
        lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
        crs_nome = src.crs.to_string() if src.crs else 'n/d'

    line = LineString(zip(lons, lats)).simplify(tolerancia, preserve_topology=True)
    if line.geom_type == 'LineString': flons, flats = line.xy
    else: flons, flats = lons, lats

    elev = terreno[pr, pc].astype(float)
    # Distancia geodesica (WGS84) entre amostras consecutivas do caminho em pixels (vizinhanca de 8):
    # independe da CRS do raster; nao ha correcao adicional de latitude por pixel.
    geod = Geod(ellps="WGS84")
    _, _, dist_step = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    dist_step = np.asarray(dist_step, dtype=float)   # pyproj pode devolver lista
    dist_acum = np.insert(np.cumsum(dist_step), 0, 0)
    _, _, d_reta = geod.inv(lons[0], lats[0], lons[-1], lats[-1])

    diam_m = hid['od_pol'] * 0.0254 - 2 * hid['esp_mm'] / 1000
    area = np.pi * (diam_m / 2)**2
    sg = 100 / ((hid['conc'] / hid['sg_solido']) + ((100 - hid['conc']) / 1.0))
    vazao = ((hid['prod'] * 1e6) / (hid['dias'] * 24 * hid['disp'])) / (hid['conc'] / 100) / sg / 3600
    vel = vazao / area
    rho_m = sg * 1000
    d = calcular_gradiente_durand(vel, diam_m, sg, hid['sg_solido'], hid['d50'], hid['rug_mm'] / 1000,
                                  hid['k_durand'], 1.0, hid['visc_cp'], hid.get('visc_polpa', False))
    j = d['J']                                   # m de polpa / m
    perda = j * dist_acum                        # m de polpa

    h1 = hid['p1_mpa'] * 1e6 / (rho_m * G)       # pressao -> carga em m de polpa
    h2 = hid['p2_mpa'] * 1e6 / (rho_m * G) if (split_point is not None and hid['p2_mpa'] > 0) else 0.0
    hgl_sem_eb2 = elev[0] + h1 - perda
    hgl = hgl_sem_eb2.copy()
    if h2 > 0:
        hgl[split_point:] += h2                  # EB-2 SOMA carga a HGL que chega (nao fixa cota+carga)
    pressao_mpa = rho_m * G * (hgl - elev) / 1e6

    desnivel = elev[-1] - elev[0]
    # Carga minima requerida = maximo ao longo do perfil de (z_i - z_0) + perda acumulada_i + folga_i,
    # com folga = regra de slack flow nos pontos intermediarios e pressao residual no terminal.
    folga_req = np.full_like(elev, float(hid['slack']), dtype=float)
    folga_req[-1] = float(hid['p_res_m'])
    h_req_perfil = (elev - elev[0]) + perda + folga_req
    i_req = int(np.argmax(h_req_perfil))
    h_req = max(0.0, float(h_req_perfil[i_req]))
    h_req_extremos = max(0.0, desnivel + perda[-1] + hid['p_res_m'])   # criterio anterior (so extremidades)
    h_inst = h1 + h2
    k_pot = rho_m * G * vazao / hid['eta'] / 1e6          # MW por m de carga
    p_adm = pressao_admissivel_barlow(hid['od_pol'] * 25.4, hid['esp_mm'], hid['smys'], hid['f_proj'])
    h_adm = p_adm * 1e6 / (rho_m * G)
    i_pmax = int(np.argmax(pressao_mpa))
    acima_maop = pressao_mpa[1:] > p_adm                      # trechos (entre amostras) acima da MAOP
    ext_acima_maop_km = float(dist_step[acima_maop].sum() / 1000)

    analise = analisar_eb2(dist_acum / 1000, elev, hgl_sem_eb2, hid['slack'], hid['p_succ_min_m'],
                           split_point if h2 > 0 else None)
    return {
        'lats': list(lats), 'lons': list(lons), 'lats_draw': list(flats), 'lons_draw': list(flons),
        'dist_arr': dist_acum/1000, 'elev': elev, 'hgl': hgl, 'hgl_sem_eb2': hgl_sem_eb2,
        'pressao_mpa': pressao_mpa, 'dist_total': dist_acum[-1]/1000,
        'potencia': k_pot * h_inst, 'potencia_min': k_pot * h_req,
        'h1': h1, 'h2': h2, 'h_inst': h_inst, 'h_req': h_req, 'perda_total_m': perda[-1],
        'h_req_extremos': h_req_extremos, 'km_h_req': float(dist_acum[i_req] / 1000),
        'km_pmax': float(dist_acum[i_pmax] / 1000), 'ext_acima_maop_km': ext_acima_maop_km,
        'Re': d['Re'], 'f_darcy': d['f'],
        'residual_terminal': hgl[-1] - elev[-1],
        'p_adm_mpa': p_adm, 'h_adm': h_adm, 'p_max_mpa': float(pressao_mpa.max()),
        'n_est_min': int(np.ceil(h_req / h_adm)),
        'p_succao_eb2_m': (float(hgl_sem_eb2[split_point] - elev[split_point]) if h2 > 0 else None),
        'p_descarga_eb2_mpa': (float(pressao_mpa[split_point]) if h2 > 0 else None),
        'analise_eb2': analise,
        'sg': sg, 'rho_m': rho_m, 'vazao_m3s': vazao, 'diam_int_m': diam_m,
        'v_op': vel, 'v_limite_deposicao': d['VL'], 'fl_durand': d['FL'], 'cv_pct': d['Cv_pct'],
        'j_gradiente': d['J'], 'j_mca': d['J_mca'],
        'regime_hidraulico': d['regime'], 'modelo_valido': d['modelo_valido'], 'aviso_modelo': d['aviso'],
        'crs': crs_nome, 'cota_origem': elev[0], 'cota_destino': elev[-1],
        'lat_o': lats[0], 'lon_o': lons[0], 'lat_d': lats[-1], 'lon_d': lons[-1],
        'dist_reta_km': d_reta/1000, 'sinuosidade': (dist_acum[-1]/d_reta if d_reta > 0 else float('nan')),
        'passo_medio_m': float(np.mean(dist_step)),
    }

@st.cache_data(show_spinner=False, hash_funcs={gpd.GeoDataFrame: lambda g: (len(g), str(g.total_bounds))})
def contar_travessias_hidrografia(lons, lats, gdf_hidro, faixa_m):
    """Numero de travessias distintas da faixa de hidrografia (APP) e extensao da rota dentro dela.
    Usa a faixa (buffer) e nao a linha do rio, para nao contar varias vezes uma mesma travessia
    quando o caminho em pixels faz zigue-zague. Calculo em EPSG:5880 (metros)."""
    if gdf_hidro is None or len(gdf_hidro) == 0:
        return None, None
    try:
        rota = gpd.GeoSeries([LineString(zip(lons, lats))], crs=4326).to_crs(5880).iloc[0]
        rios = gdf_hidro.to_crs(5880).geometry
        faixa = rios.buffer(max(faixa_m, 1.0), resolution=4).unary_union
        inter = rota.intersection(faixa)
        if inter.is_empty: return 0, 0.0
        partes = list(inter.geoms) if hasattr(inter, 'geoms') else [inter]
        partes = [p for p in partes if p.length > 0]
        return len(partes), float(sum(p.length for p in partes) / 1000)
    except Exception:
        return None, None

hid = dict(prod=prod, conc=conc, dias=dias_op, disp=disp, od_pol=od_pol, esp_mm=esp_mm, sg_solido=sg_solido,
           d50=d50, rug_mm=rugosidade_mm, visc_cp=visc_cp, visc_polpa=visc_polpa, k_durand=k_durand, eta=eta, p1_mpa=p1_mpa,
           p2_mpa=p2_mpa if usar_eb2 else 0.0, p_res_m=p_res_m, slack=regra_slack_flow,
           p_succ_min_m=p_succ_min_m, smys=smys_mpa, f_proj=f_proj)
def _resolver_eb2_auto(rota_, hid_):
    """Posiciona a EB-2 automaticamente sobre a propria rota (que ja so tem EB-1) e, se pedido, dimensiona o incremento."""
    r0 = processar_completo(rota_, tol_geo, None, tiff_path, fator, terreno, dict(hid_, p2_mpa=0.0))
    if r0 is None: return None, dict(hid_, p2_mpa=0.0), None
    a = r0['analise_eb2']
    if not a['necessaria']:
        return None, dict(hid_, p2_mpa=0.0), "Com a EB-1 a folga é respeitada em todo o traçado: EB-2 não necessária."
    if a['idx_sugerido'] is None:
        return None, dict(hid_, p2_mpa=0.0), ("Há violação de folga, mas não existe posição a montante com folga de sucção suficiente. "
                                              "Aumente a pressão da EB-1 ou reduza a pressão mínima de sucção.")
    s = a['idx_sugerido']
    if h2_auto_dim:
        p2 = float(np.ceil(a['h2_min_sugerido'] * r0['rho_m'] * G / 1e6 * 100) / 100)
    else:
        p2 = hid_['p2_mpa']
    return s, dict(hid_, p2_mpa=p2), None

hid_n, hid_g = hid, hid
aviso_eb2_auto = None
if eb2_auto:
    split_n, hid_n, aviso_eb2_auto = _resolver_eb2_auto(rota_n, hid)
    split_g, hid_g, _ = _resolver_eb2_auto(rota_g, hid)
    hid = hid_n   # relatorio/inputs refletem a rota otimizada
res_n = processar_completo(rota_n, tol_geo, split_n, tiff_path, fator, terreno, hid_n)
res_g = processar_completo(rota_g, tol_geo, split_g, tiff_path, fator, terreno, hid_g)
if res_n is not None:
    _n_trav, _ext_trav = contar_travessias_hidrografia(tuple(res_n['lons']), tuple(res_n['lats']),
                                                       gdf_hidro_carregado, buffer_hidrografia)
    res_n['travessias_hidro'], res_n['ext_em_app_km'] = _n_trav, _ext_trav

if rota_n is not None and res_n is not None:
    _decl_rota = declividade[rota_n[:, 0], rota_n[:, 1]]
    res_n['decl_max_rota'] = float(np.nanmax(_decl_rota))
    res_n['cel_acima_rampa'] = int(_masc_rampa[rota_n[:, 0], rota_n[:, 1]].sum())
    res_n['cel_rota'] = int(len(rota_n))
    res_n['cota_max_rota'] = float(np.nanmax(res_n['elev']))

inputs_rep = dict(hid, limitar_rampa=limitar_rampa, rampa_max_pct=rampa_max_pct, custo_rampa=custo_rampa, pesos_camadas=pesos_camadas, peso_hidro=CUSTO_RESTRICAO_HIDROGRAFIA,
                  peso_desenho=CUSTO_RESTRICAO_DESENHO_MANUAL, eb2=usar_eb2 and (split_n is not None), slack=regra_slack_flow, peso=peso_declive, tol_geo=tol_geo,
                  app_m=buffer_hidrografia, fator=fator, pixel_size=pixel_size,
                  tiff_name=os.path.basename(tiff_path))

# --- 7. TABS DE VISUALIZAÇÃO ---

with tab1:
    if eb2_auto and aviso_eb2_auto: st.warning(aviso_eb2_auto)
    elif eb2_auto and res_n is not None and split_n is not None:
        st.success(f"EB-2 posicionada automaticamente no km {res_n['dist_arr'][split_n]:.2f} (incremento {hid['p2_mpa']:.2f} MPa).")
    st.markdown("**Instruções:** Desenhe polígonos no mapa para criar restrições temporárias. A rota recalculará automaticamente desviando da área.")
    
    with rasterio.open(tiff_path) as src:
        lats_m, lons_m = pixel_latlon(src, [0, src.height], [0, src.width])
        bounds_latlon = [[min(lats_m), min(lons_m)], [max(lats_m), max(lons_m)]]
    
    m = folium.Map(location=[(bounds_latlon[0][0]+bounds_latlon[1][0])/2, (bounds_latlon[0][1]+bounds_latlon[1][1])/2], zoom_start=10, max_bounds=True,
                   tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')
    m.fit_bounds(bounds_latlon)
    Draw(export=True, position='topleft', draw_options={'polyline':False, 'circlemarker':False, 'circle':False}).add_to(m)
    
    if st.session_state.get('user_drawings'):
        fc = {"type": "FeatureCollection", "features": st.session_state['user_drawings']}
        folium.GeoJson(fc, style_function=lambda x: {'color': 'red', 'fillColor': 'red', 'weight': 2}).add_to(m)
    
    for gdf in gdfs_carregados:
        if gdf.crs != "EPSG:4326":
            gdf_mapa = gdf.to_crs("EPSG:4326")
        else:
            gdf_mapa = gdf
        gdf_mapa = _geo_para_mapa(gdf_mapa)
        
        folium.GeoJson(
            gdf_mapa, 
            style_function=lambda x: {'color': 'orange', 'fillColor': 'orange', 'fillOpacity': 0.4, 'weight': 2},
            name="Restrição Zip"
        ).add_to(m)

    if gdf_hidro_carregado is not None:
        gdf_hidro_mapa = gdf_hidro_carregado if gdf_hidro_carregado.crs == "EPSG:4326" else gdf_hidro_carregado.to_crs("EPSG:4326")
        gdf_hidro_mapa = _geo_para_mapa(gdf_hidro_mapa)
        folium.GeoJson(
            gdf_hidro_mapa,
            style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.3, 'weight': 2},
            name="Hidrografia / APP"
        ).add_to(m)

    if res_g: folium.PolyLine(list(zip(res_g['lats_draw'], res_g['lons_draw'])), color="purple", weight=3, dash_array='5, 5', tooltip="Rota por Gravidade").add_to(m)
    if res_n: folium.PolyLine(list(zip(res_n['lats_draw'], res_n['lons_draw'])), color="red", weight=4, tooltip="Rota Otimizada").add_to(m)
    
    def add_marker(coords, cor, nome, icone):
        with rasterio.open(tiff_path) as src:
            la, lo = pixel_latlon(src, *src.index(*coords))
            folium.Marker([la[0], lo[0]], popup=nome, icon=folium.Icon(color=cor, icon=icone)).add_to(m)
            
    if st.session_state['inicio_coords']: add_marker(st.session_state['inicio_coords'], 'green', 'Origem', 'play')
    if st.session_state['fim_coords']: add_marker(st.session_state['fim_coords'], 'blue', 'Destino', 'stop')
    if eb2_manual and st.session_state['eb2_coords']: add_marker(st.session_state['eb2_coords'], 'orange', 'EB-2', 'flag')
    if eb2_auto and res_n is not None and split_n is not None:
        folium.Marker([res_n['lats'][split_n], res_n['lons'][split_n]], icon=folium.Icon(color='orange', icon='flag'),
                      popup=f"EB-2 (automática) km {res_n['dist_arr'][split_n]:.2f}").add_to(m)

    map_data = st_folium(m, width="100%", height=600)
    
    if map_data and map_data.get('all_drawings') is not None:
        current_drawings = map_data['all_drawings']
        if current_drawings != st.session_state['user_drawings']:
            st.session_state['user_drawings'] = current_drawings
            st.rerun() 

    if map_data and map_data.get('last_clicked'):
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        with rasterio.open(tiff_path) as src:
            x, y = latlon_utm(src, lat, lon)
            if modo == "🟢 Origem": st.session_state['inicio_coords'] = (x, y); st.rerun()
            elif modo == "🔵 Destino": st.session_state['fim_coords'] = (x, y); st.rerun()
            elif modo == "📍 EB-2": st.session_state['eb2_coords'] = (x, y); st.rerun()

with tab2:
    if res_n:
        mpa_m = lambda h: h * res_n['rho_m'] * 9.81 / 1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['elev'], fill='tozeroy', line=dict(color='red', width=1), name='Terreno (Otimizada)'))
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['hgl'], line=dict(color='blue', width=2), name='HGL (Otimizada)'))
        if usar_eb2 and res_n['h2'] > 0:
            fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['hgl_sem_eb2'], line=dict(color='gray', width=1, dash='dot'), name='HGL só com EB-1'))
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['elev']+regra_slack_flow, line=dict(color='orange', dash='dot'), name=f'Limite de Slack Flow (+{regra_slack_flow} m)'))
        if res_g:
            fig.add_trace(go.Scatter(x=res_g['dist_arr'], y=res_g['elev'], line=dict(color='purple', width=1, dash='dash'), name='Terreno (Gravidade)', visible='legendonly'))
            fig.add_trace(go.Scatter(x=res_g['dist_arr'], y=res_g['hgl'], line=dict(color='lightblue', width=2, dash='dash'), name='HGL (Gravidade)', visible='legendonly'))
        fig.update_layout(title="Perfil Piezométrico (m de polpa) e Análise de Slack Flow", xaxis_title="km", yaxis_title="Cota / HGL (m de polpa)", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        figp = go.Figure()
        figp.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['pressao_mpa'], line=dict(color='blue'), name='Pressão interna'))
        figp.add_hline(y=res_n['p_adm_mpa'], line_dash='dash', line_color='red', annotation_text=f"MAOP (Barlow) = {res_n['p_adm_mpa']:.1f} MPa")
        figp.update_layout(title="Pressão interna ao longo do duto", xaxis_title="km", yaxis_title="MPa", height=350, template="plotly_white")
        st.plotly_chart(figp, use_container_width=True)

        if not res_n.get('modelo_valido', True):
            st.warning(f"⚠️ **Modelo fora do domínio de validade:** {res_n['aviso_modelo']}")
        elif res_n.get('regime_hidraulico') == 'homogeneo':
            st.info(f"ℹ️ {res_n['aviso_modelo']}")

        c1, c2, c3, c4 = st.columns(4)
        margem = res_n['hgl'] - res_n['elev']
        c1.metric("Velocidade Operação", f"{res_n['v_op']:.2f} m/s")
        c2.metric("Menor Folga HGL-terreno", f"{margem.min():.1f} m")
        c3.metric("Vel. Limite (VL)", f"{res_n['v_limite_deposicao']:.2f} m/s")
        c4.metric("Pressão máx. / MAOP", f"{res_n['p_max_mpa']:.1f} / {res_n['p_adm_mpa']:.1f} MPa")

        trechos = trechos_criticos(res_n['dist_arr'], margem, regra_slack_flow)
        st.caption(f"Intervalo médio de amostragem: {res_n['passo_medio_m']:.0f} m | tolerância de violação: {TOL_VIOLACAO_M} m")
        if not trechos:
            st.success(f"✅ HGL ≥ {regra_slack_flow} m acima do terreno em todo o traçado.")
        else:
            st.error(f"❌ Trechos com folga < {regra_slack_flow} m (Slack Flow). Aumente a carga ou posicione/dimensione a EB-2.")
            st.dataframe(pd.DataFrame(trechos).rename(columns={'ini': 'Início (km)', 'fim': 'Fim (km)', 'ext': 'Extensão (km)', 'p_min': 'Folga mínima (m)', 'n': 'Amostras'}), use_container_width=True)

        if res_n['p_max_mpa'] > res_n['p_adm_mpa']:
            st.error(f"❌ Pressão máxima ({res_n['p_max_mpa']:.2f} MPa) excede a MAOP ({res_n['p_adm_mpa']:.2f} MPa). "
                     f"Reduza a carga por estação, use estação de válvulas/choke antes da EB-2 ou aumente a espessura.")
        st.info(f"Carga requerida (atrito + desnível + residual): {res_n['h_req']:.0f} m de polpa ({mpa_m(res_n['h_req']):.1f} MPa). "
                f"Carga máxima por estação (MAOP): {res_n['h_adm']:.0f} m ⇒ no mínimo {res_n['n_est_min']} estação(ões).")
        st.write(f"**Potência de eixo** (η = {eta:.2f}): instalada = {res_n['potencia']:.2f} MW "
                 f"({'EB-1 + EB-2' if res_n['h2'] > 0 else 'somente EB-1'}, H = {res_n['h_inst']:.0f} m de polpa); "
                 f"mínima requerida = {res_n['potencia_min']:.2f} MW (H = {res_n['h_req']:.0f} m).")
        st.write(f"**Residual no terminal:** {res_n['residual_terminal']:.0f} m de polpa "
                 f"(excedente a dissipar em estação de válvulas ≈ {max(0, res_n['residual_terminal']-p_res_m):.0f} m).")
        st.write(f"**Pressão máxima** de {res_n['p_max_mpa']:.2f} MPa no km {res_n['km_pmax']:.1f}; "
                 f"extensão acima da MAOP: {res_n['ext_acima_maop_km']:.1f} km. "
                 f"**H_req** governado pelo km {res_n['km_h_req']:.1f} (critério só das extremidades: {res_n['h_req_extremos']:.0f} m).")
        if res_n.get('decl_max_rota') is not None:
            st.write(f"**Rampa:** declividade máxima do terreno na rota = {100*res_n['decl_max_rota']:.1f}%; cota máxima = "
                     f"{res_n['cota_max_rota']:.0f} m; células da rota acima do limite"
                     f"{f' de {rampa_max_pct:.0f}%' if limitar_rampa else ' (sem limite ativo)'}: "
                     f"{res_n['cel_acima_rampa']} de {res_n['cel_rota']}.")
        if res_n.get('travessias_hidro') is not None:
            st.write(f"**Hidrografia:** {res_n['travessias_hidro']} travessias da faixa de {buffer_hidrografia} m; "
                     f"{res_n['ext_em_app_km']:.1f} km da rota dentro da faixa.")
        if st.session_state.get('tempo_calc_s') is not None:
            st.caption(f"Tempo da última execução completa (custos + rotas): {st.session_state['tempo_calc_s']:.0f} s "
                       f"(execuções em cache são mais rápidas). Registre o computador e o número de execuções.")
        with st.expander("⏱️ Tempo por etapa e configuração do computador"):
            _et = st.session_state.get('tempos_etapas') or []
            if _et:
                st.dataframe(pd.DataFrame([{"Etapa": e, "Tempo (s)": round(t, 2), "Feições": n} for e, t, n in _et]),
                             hide_index=True, use_container_width=True)
                st.caption(f"Células do grafo: {st.session_state.get('n_celulas_grafo', 0):,}. "
                           "Etapas já em cache aparecem com tempo próximo de zero: para medir de verdade, limpe o cache e rode de novo.")
            if st.button("🧹 Limpar cache e recalcular"):
                st.cache_data.clear(); st.rerun()
            import platform
            _cfg = f"{platform.system()} {platform.release()}; {platform.processor() or platform.machine()}; {os.cpu_count()} núcleos lógicos; Python {platform.python_version()}"
            try:
                import psutil
                _cfg += f"; RAM {psutil.virtual_memory().total/2**30:.1f} GB"
            except Exception:
                pass
            st.code(_cfg)
            st.caption("Copie esta linha para a seção de métodos (configuração do computador).")
        with st.expander("Análise de sensibilidade (viscosidade aparente e peso de declividade)"):
            visc_lista = st.text_input("Viscosidades aparentes da polpa (cP), separadas por vírgula", "5, 8.85, 10, 20")
            st.caption("A 1ª linha repete o caso base (mesma viscosidade e mesmo Reynolds do memorial). Nas demais, a viscosidade é a "
                       "APARENTE da polpa e o Reynolds usa a massa específica da polpa (Re = ρm·V·D/μ).")
            linhas = [{'Caso': f"Base ({visc_cp:g} cP, {'polpa' if visc_polpa else 'carreador'})", 'Visc. (cP)': visc_cp,
                       'Re': f"{res_n['Re']:.3g}", 'f': round(res_n['f_darcy'], 5), 'J (m/m)': round(res_n['j_gradiente'], 6),
                       'Perda total (m)': round(float(res_n['perda_total_m'])), 'H_req (m)': round(res_n['h_req']),
                       'N mín.': int(np.ceil(res_n['h_req'] / res_n['h_adm'])),
                       'Potência mín. (MW)': round(res_n['rho_m'] * G * res_n['vazao_m3s'] / eta / 1e6 * res_n['h_req'], 2)}]
            try:
                for vcp in [float(x) for x in visc_lista.split(',') if x.strip()]:
                    dd = calcular_gradiente_durand(res_n['v_op'], res_n['diam_int_m'], res_n['sg'], sg_solido, d50,
                                                   rugosidade_mm / 1000, k_durand, 1.0, vcp, True)
                    perda_s = dd['J'] * res_n['dist_arr'] * 1000
                    fol = np.full_like(res_n['elev'], float(regra_slack_flow), dtype=float); fol[-1] = p_res_m
                    hreq_s = max(0.0, float(np.max((res_n['elev'] - res_n['elev'][0]) + perda_s + fol)))
                    linhas.append({'Caso': 'Aparente (polpa)', 'Visc. (cP)': vcp, 'Re': f"{dd['Re']:.3g}", 'f': round(dd['f'], 5),
                                   'J (m/m)': round(dd['J'], 6), 'Perda total (m)': round(float(perda_s[-1])),
                                   'H_req (m)': round(hreq_s), 'N mín.': int(np.ceil(hreq_s / res_n['h_adm'])),
                                   'Potência mín. (MW)': round(res_n['rho_m'] * G * res_n['vazao_m3s'] / eta / 1e6 * hreq_s, 2)})
                st.dataframe(pd.DataFrame(linhas), use_container_width=True)
            except Exception as e:
                st.error(f"Erro na sensibilidade à viscosidade: {e}")
            st.markdown("**Peso de declividade** (recalcula a rota para cada peso; pode demorar)")
            pesos_txt = st.text_input("Pesos", "1, 2.5, 5, 10, 20")
            if st.button("Rodar sensibilidade ao peso"):
                penal = custo_base - (declividade * peso_declive + 1.0)
                geod_s = Geod(ellps="WGS84"); linhas_p = []
                for w in [float(x) for x in pesos_txt.split(',') if x.strip()]:
                    rota_w, _ = calcular_segmentos(pontos_idx, declividade * w + 1.0 + penal)
                    if rota_w is None: continue
                    with rasterio.open(tiff_path) as src_w:
                        la_w, lo_w = pixel_latlon(src_w, rota_w[:, 0]*fator, rota_w[:, 1]*fator)
                    _, _, dst_w = geod_s.inv(lo_w[:-1], la_w[:-1], lo_w[1:], la_w[1:])
                    dst_w = np.asarray(dst_w, dtype=float)
                    linhas_p.append({'Peso': w, 'Extensão (km)': round(float(dst_w.sum()/1000), 2),
                                     'Sinuosidade': round(float(dst_w.sum()/ (res_n['dist_reta_km']*1000)), 3),
                                     'Declividade média na rota (m/m)': round(float(declividade[rota_w[:,0], rota_w[:,1]].mean()), 4),
                                     'Declividade máx. (%)': round(100*float(declividade[rota_w[:,0], rota_w[:,1]].max()), 1),
                                     'Cota máx. (m)': round(float(terreno[rota_w[:,0], rota_w[:,1]].max())),
                                     'Células acima da rampa': int(_masc_rampa[rota_w[:,0], rota_w[:,1]].sum())})
                st.dataframe(pd.DataFrame(linhas_p), use_container_width=True)

        a = res_n['analise_eb2']
        if a['necessaria']:
            msg = f"**EB-2:** com somente a EB-1, a 1ª violação ocorre no km {a['km_primeira_violacao']:.2f}. "
            if a['km_sugerido'] is not None:
                msg += (f"Posição sugerida (última com folga ≥ {regra_slack_flow} + {p_succ_min_m} m): **km {a['km_sugerido']:.2f}**; "
                        f"carga mínima: {a['h2_min_sugerido']:.0f} m de polpa ({mpa_m(a['h2_min_sugerido']):.2f} MPa). ")
            else:
                msg += "Não há posição a montante com folga de sucção suficiente. "
            if a['h2_min_usuario'] is not None:
                msg += f"Carga mínima para a EB-2 na posição escolhida: {a['h2_min_usuario']:.0f} m ({mpa_m(a['h2_min_usuario']):.2f} MPa)."
            st.info(msg)
        if usar_eb2 and res_n['p_succao_eb2_m'] is not None:
            if res_n['p_succao_eb2_m'] < regra_slack_flow + p_succ_min_m:
                st.error(f"❌ Sucção da EB-2 com {res_n['p_succao_eb2_m']:.0f} m de polpa, abaixo do mínimo ({regra_slack_flow + p_succ_min_m} m). Mova a EB-2 para montante.")
            st.caption(f"Pressão de descarga da EB-2 = {res_n['p_descarga_eb2_mpa']:.2f} MPa (MAOP {res_n['p_adm_mpa']:.2f} MPa).")
            if res_n['p_descarga_eb2_mpa'] > res_n['p_adm_mpa']:
                st.error(f"❌ A descarga da EB-2 ({res_n['p_descarga_eb2_mpa']:.2f} MPa) excede a MAOP ({res_n['p_adm_mpa']:.2f} MPa). "
                         f"Reduza o incremento, use parede mais espessa a jusante ou acrescente uma estação.")

        if res_n.get('regime_hidraulico') == 'homogeneo':
            st.caption("Regime pseudo-homogêneo: VL/FL de Durand são apenas referência e não bloqueiam a rota.")
        elif res_n['v_op'] < res_n['v_limite_deposicao']:
            st.error(f"❌ V ({res_n['v_op']:.2f} m/s) abaixo de VL ({res_n['v_limite_deposicao']:.2f} m/s): risco de leito de sólidos.")
        else:
            st.success("✅ V acima da VL de Durand.")

with tab3:
    st.subheader("📥 Exportação de Resultados")
    if res_n:
        c1, c2 = st.columns(2)
        df_rota = pd.DataFrame({
            'Distancia_km': res_n['dist_arr'], 'Latitude': res_n['lats'], 'Longitude': res_n['lons'],
            'Elevacao_terreno_m': res_n['elev'], 'HGL_m_polpa': res_n['hgl'], 'Pressao_MPa': res_n['pressao_mpa']
        })
        csv = df_rota.to_csv(index=False).encode('utf-8')
        df_param = pd.DataFrame(montar_parametros(inputs_rep, res_n), columns=['Parâmetro', 'Valor', 'Observação'])
        with c1:
            st.markdown("#### Planilha de Coordenadas")
            st.download_button("📊 Baixar Rota Otimizada (.CSV)", data=csv, file_name='rota_mineroduto.csv', mime='text/csv', use_container_width=True)
            st.download_button("📋 Baixar Parâmetros (Tabela 1) (.CSV)", data=df_param.to_csv(index=False).encode('utf-8'), file_name='parametros_simulacao.csv', mime='text/csv', use_container_width=True)
        with c2:
            st.markdown("#### Memorial Descritivo")
            st.write("Gere o memorial com premissas, equações, resultados e verificações.")
            if st.button("🖨️ Baixar Memorial de Cálculo (.PDF)", use_container_width=True):
                with st.spinner("Gerando PDF..."):
                    pdf_data = gerar_pdf_bytes(inputs_rep, res_n, res_g, fator, tiff_path, sid)
                    st.download_button("Clique aqui para salvar o PDF", data=pdf_data, file_name='Memorial_Calculo_TCC.pdf', mime='application/pdf', use_container_width=True)
                    st.success("Memorial PDF gerado com sucesso!")
        st.markdown("---")
        st.write("Parâmetros de entrada e geometria (Tabela 1 do artigo):")
        st.dataframe(df_param, use_container_width=True)
        st.write("Visualização rápida dos dados (CSV):")
        st.dataframe(df_rota.head(10), use_container_width=True)
    else:
        st.info("Calcule uma rota nas abas anteriores primeiro para visualizar e exportar os dados.")
