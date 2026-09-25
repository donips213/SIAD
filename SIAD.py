import streamlit as st
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
import matplotlib.pyplot as plt

# --- 0. CONSTANTES NOMEADAS (evita "numeros magicos" espalhados no codigo) ---
CUSTO_AREA_INVALIDA = 10_000_000.0       # penalidade para pixels sem dado de terreno valido
CUSTO_RESTRICAO_SHAPEFILE = 500_000.0    # penalidade por trecho dentro de shapefile de restricao
CUSTO_RESTRICAO_DESENHO_MANUAL = 5_000_000.0  # penalidade por trecho dentro de area desenhada no mapa
CUSTO_PENALIDADE_GRAVIDADE = 1_000_000.0      # penalidade para pontos acima da cota da inicio (rota gravidade)
CUSTO_RESTRICAO_HIDROGRAFIA = 500_000.0       # penalidade por trecho dentro da faixa de rios/corpos d'agua (APP)

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

def gerar_imagens_temp(res_n, res_g, fator, tiff_path, sid, regra_slack=20):
    path_perfil = f"temp_perfil_{sid}.png"
    path_mapa = f"temp_mapa_{sid}.png"
    plt.figure(figsize=(10, 4))
    if res_n:
        plt.plot(res_n['dist_arr'], res_n['elev'], color='red', alpha=0.3, label='Terreno (Otimizada)')
        plt.plot(res_n['dist_arr'], res_n['hgl'], color='blue', label='HGL (Otimizada)')
        plt.plot(res_n['dist_arr'], res_n['elev'] + regra_slack, color='orange', alpha=0.5, linestyle=':', label=f'Limite Slack Flow (+{regra_slack}m)')
    if res_g:
        plt.plot(res_g['dist_arr'], res_g['elev'], color='purple', alpha=0.3, linestyle='--', label='Terreno (Gravidade)')
        plt.plot(res_g['dist_arr'], res_g['hgl'], color='lightblue', linestyle='--', label='HGL (Gravidade)')
    plt.title("Perfil Hidraulico Comparativo")
    plt.xlabel("Distancia (km)")
    plt.ylabel("Elevacao (m.c.a)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_perfil, dpi=100)
    plt.close()

    try:
        with rasterio.open(tiff_path) as src:
            new_shape = (int(src.height/fator), int(src.width/fator))
            data = src.read(1, out_shape=new_shape)
            plt.figure(figsize=(8, 8))
            plt.imshow(data, cmap='terrain', alpha=0.8)
        
        if res_n:
            xs, ys = transform('EPSG:4326', src.crs, res_n['lons_draw'], res_n['lats_draw'])
            # Solução: iteração ponto a ponto em vez de array direto
            rows_cols = [src.index(x, y) for x, y in zip(xs, ys)]
            rows = [rc[0] for rc in rows_cols]
            cols = [rc[1] for rc in rows_cols]
            plt.plot(np.array(cols)/fator, np.array(rows)/fator, 'r-', linewidth=2, label='Otimizada')
            
        if res_g:
            xs, ys = transform('EPSG:4326', src.crs, res_g['lons_draw'], res_g['lats_draw'])
            # Solução: iteração ponto a ponto em vez de array direto
            rows_cols = [src.index(x, y) for x, y in zip(xs, ys)]
            rows = [rc[0] for rc in rows_cols]
            cols = [rc[1] for rc in rows_cols]
            plt.plot(np.array(cols)/fator, np.array(rows)/fator, 'm--', linewidth=2, label='Gravidade')
            
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
    except Exception as e:
        st.warning(f"Não foi possível gerar a imagem do mapa para o PDF: {e}")
    return path_perfil, path_mapa

def gerar_pdf_bytes(inputs, res_n, res_g, fator, tiff_path, sid):
    regra_slack = inputs['slack_flow']
    path_perfil, path_mapa = gerar_imagens_temp(res_n, res_g, fator, tiff_path, sid, regra_slack)
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title('1. DADOS DE ENTRADA DO PROJETO')
    col_w = 45
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(col_w, 6, "Parametro", 1)
    pdf.cell(col_w, 6, "Valor Adotado", 1)
    pdf.cell(col_w*2, 6, "Referencia / Observacao", 1)
    pdf.ln()
    pdf.set_font('Arial', '', 9)
    dados = [
        ("Producao", f"{inputs['prod']} Mtpa", "Define a vazao do sistema"),
        ("Diametro Interno", f"{inputs['diam']} pol", "Tubulacao principal API 5L"),
        ("Concentracao (Cw)", f"{inputs['conc']} %", "Concentracao de solidos em peso"),
        ("Carga na inicio", f"{inputs['carga_inicio']} m.c.a", "Pressao de impulsao inicial"),
        ("Regra Slack Flow", f"{regra_slack} m.c.a", "Folga minima de seguranca Anti-Cavitacao"),
        ("EB-2 Intermediaria", "Sim" if inputs['eb2'] else "Nao", f"Carga EB-2: {inputs['carga_eb2']} m.c.a")
    ]
    for row in dados:
        pdf.cell(col_w, 6, pdf_safe(row[0]), 1)
        pdf.cell(col_w, 6, pdf_safe(row[1]), 1)
        pdf.cell(col_w*2, 6, pdf_safe(row[2]), 1)
        pdf.ln()
    pdf.ln(5)

    # --- NOVO: FÓRMULAS E EQUAÇÕES ---
    pdf.chapter_title('2. MODELO FISICO E EQUACOES UTILIZADAS')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, "O dimensionamento do mineroduto utiliza as seguintes equacoes matematicas para modelar o escoamento hidraulico:")
    pdf.ln(2)
    
    pdf.set_font('Courier', '', 8)
    formulas = (
        "1. Gravidade Especifica da Mistura (SG):\n"
        "   SG = 100 / [ (Cw / SG_Solido) + ((100 - Cw) / SG_Agua) ]\n\n"
        "2. Vazao Volumetrica (Q) em m3/s:\n"
        "   Q = Producao_Anual / (Dias * 24h * Disp) / (Cw * SG * 3600)\n\n"
        "3. Velocidade de Operacao (V) em m/s:\n"
        "   V = Q / Area_Interna\n\n"
        "4. Perda de Carga do Fluido Carreador (Darcy-Weisbach + Swamee-Jain):\n"
        "   f = 0.25 / [log10(k/(3.7D) + 5.74/Re^0.9)]^2\n"
        "   i_agua = f * V^2 / (2 * g * D)\n\n"
        "5. Fator de Durand (FL) e Velocidade Limite de Deposicao (VL):\n"
        "   FL = 0.4794 + 0.5429*(0.01*Cv)^0.1058*(log10(d50) - 1)\n"
        "   VL = FL * sqrt( 2 * g * D * (SG_Solido - 1) )\n\n"
        "6. Correlacao de Durand-Condolios (1952) - efeito dos solidos:\n"
        "   Phi = K * [ V^2 / (g * D * (SG_Solido - 1)) ]^-1.5\n"
        "   J = i_agua * (1 + Cv_fracao * Phi)\n\n"
        "7. Linha de Gradiente Hidraulico (HGL) em m.c.a:\n"
        "   HGL[i] = Elevacao_Inicial + Carga_Bomba - (J * Distancia_Acumulada[i])\n\n"
        "8. Potencia de Eixo Requerida (Pot) em MW:\n"
        "   Pot = (SG * 1000 * 9.81 * Q * Altura_Manometrica_Total) / (Rendimento * 1e6)"
    )
    pdf.multi_cell(0, 4, formulas)
    pdf.ln(5)

    pdf.chapter_title('3. MEMORIAL DE CALCULOS HIDRAULICOS (ROTA OTIMIZADA)')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "Resultados diretos da aplicacao das equacoes na malha topografica da rota otimizada:")
    pdf.ln(2)
    
    cw = inputs['conc'] / 100
    sg = 100 / ((inputs['conc']/inputs['sg_solido']) + ((100-inputs['conc'])/1.0))
    vazao_m3s = ((inputs['prod']*1e6)/(365*24*0.92)) / cw / sg / 3600
    vazao_m3h = vazao_m3s * 3600
    j_gradiente = res_n['j_gradiente']
    
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(90, 6, "Variavel Hidraulica", 1)
    pdf.cell(100, 6, "Valor Calculado", 1)
    pdf.ln()
    pdf.set_font('Arial', '', 9)
    
    risco_deposicao = "SIM - Risco de Sedimentacao" if res_n['v_op'] < res_n['v_limite_deposicao'] else "Nao (V > VL)"
    regime_label = "Pseudo-homogeneo (particula fina)" if res_n.get('regime_hidraulico') == 'homogeneo' else "Heterogeneo (Durand-Condolios)"
    calc_dados = [
        ("Gravidade Especifica (SG)", f"{sg:.2f} t/m3"),
        ("Vazao Volumetrica", f"{vazao_m3h:.2f} m3/h  ({vazao_m3s:.4f} m3/s)"),
        ("Concentracao Volumetrica (Cv)", f"{res_n['cv_pct']:.2f} %"),
        ("Velocidade de Operacao (V)", f"{res_n['v_op']:.2f} m/s"),
        ("Regime de Escoamento Adotado", regime_label),
        ("Fator de Durand (FL)", f"{res_n['fl_durand']:.3f}"),
        ("Velocidade Limite de Deposicao (VL)", f"{res_n['v_limite_deposicao']:.2f} m/s"),
        ("Risco de Sedimentacao (V < VL)", risco_deposicao),
        ("Perda de Carga Unitaria (J)", f"{j_gradiente:.6f} m/m"),
        ("Extensao Total da Rota", f"{res_n['dist_total']:.2f} km"),
        ("Potencia de Eixo Estimada", f"{res_n['potencia']:.2f} MW")
    ]
    for row in calc_dados:
        pdf.cell(90, 6, pdf_safe(row[0]), 1)
        pdf.cell(100, 6, pdf_safe(row[1]), 1)
        pdf.ln()

    if not res_n.get('modelo_valido', True) or res_n.get('aviso_modelo'):
        pdf.ln(3)
        pdf.set_text_color(200, 0, 0) if not res_n.get('modelo_valido', True) else pdf.set_text_color(0, 0, 150)
        pdf.set_font('Arial', 'I', 8)
        pdf.multi_cell(0, 4.5, f"Nota sobre o modelo hidraulico: {res_n.get('aviso_modelo', '')}")
        pdf.set_text_color(0, 0, 0)

    pdf.ln(5)
    pdf.chapter_title('4. ANALISE ANTI-CAVITACAO E RISCOS DE PRESSAO')
    
    slack_indices = np.where(res_n['hgl'] < (res_n['elev'] + regra_slack))[0]
    
    if len(slack_indices) > 0:
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 5, f"ALERTA: Trechos onde a pressao cai abaixo de {regra_slack} m.c.a, havendo risco critico de cavitacao (Slack Flow). Aumente a Carga da Bomba ou insira uma EB-2:")
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(40, 5, "Inicio (km)", 1)
        pdf.cell(40, 5, "Fim (km)", 1)
        pdf.cell(40, 5, "Extensao (km)", 1)
        pdf.cell(70, 5, "Pressao Minima Atingida", 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 8)
        dist_arr = res_n['dist_arr']
        diffs = np.diff(slack_indices)
        breaks = np.where(diffs > 1)[0]
        start_idx = 0
        for bk in breaks:
            grupo = slack_indices[start_idx : bk+1]
            km_ini = dist_arr[grupo[0]]
            km_fim = dist_arr[grupo[-1]]
            pressao_min = np.min(res_n['hgl'][grupo] - res_n['elev'][grupo])
            pdf.cell(40, 5, f"{km_ini:.2f}", 1)
            pdf.cell(40, 5, f"{km_fim:.2f}", 1)
            pdf.cell(40, 5, f"{(km_fim - km_ini):.2f}", 1)
            pdf.cell(70, 5, f"{pressao_min:.2f} m.c.a", 1)
            pdf.ln()
            start_idx = bk + 1
        grupo = slack_indices[start_idx:]
        if len(grupo) > 0:
            km_ini = dist_arr[grupo[0]]
            km_fim = dist_arr[grupo[-1]]
            pressao_min = np.min(res_n['hgl'][grupo] - res_n['elev'][grupo])
            pdf.cell(40, 5, f"{km_ini:.2f}", 1)
            pdf.cell(40, 5, f"{km_fim:.2f}", 1)
            pdf.cell(40, 5, f"{(km_fim - km_ini):.2f}", 1)
            pdf.cell(70, 5, f"{pressao_min:.2f} m.c.a", 1)
            pdf.ln()
    else:
        pdf.set_text_color(0, 100, 0) 
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, f"SUCESSO: A Linha Piezometrica (HGL) mantem-se pelo menos {regra_slack}m acima do terreno em todo o tracado. Sem risco de Slack Flow e bolhas de vapor.")
        pdf.set_text_color(0, 0, 0)

    pdf.add_page()
    pdf.chapter_title('5. ANEXOS GRAFICOS')
    if os.path.exists(path_perfil):
        pdf.ln(5)
        pdf.cell(0, 10, "Figura 1: Perfil Topografico e Hidraulico", 0, 1, 'C')
        pdf.image(path_perfil, x=10, w=190)
    if os.path.exists(path_mapa):
        pdf.ln(10)
        pdf.cell(0, 10, "Figura 2: Tracado em Planta e Matriz de Friccao", 0, 1, 'C')
        pdf.image(path_mapa, x=20, w=170)
    try:
        if os.path.exists(path_perfil): os.remove(path_perfil)
        if os.path.exists(path_mapa): os.remove(path_mapa)
    except OSError as e:
        st.warning(f"Não foi possível remover arquivos temporários: {e}")
    return bytes(pdf.output())

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
    xs, ys = transform({'init': 'EPSG:4326'}, src.crs, [lon], [lat])
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
            zip_path = f"zip://{os.path.join(tmp, 'temp.zip')}"

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
tiff_file = st.sidebar.file_uploader("Upload TIFF", type=["tif", "tiff"])
if tiff_file:
    tiff_path = f"temp_topografia_{sid}.tif"
    with open(tiff_path, "wb") as f: f.write(tiff_file.getbuffer())
elif os.path.exists("topografia_utm.tif"): 
    tiff_path = "topografia_utm.tif"
else: 
    st.warning("Faça o upload do TIFF para iniciar.")
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
tol_geo = st.sidebar.slider("Simplificação (Graus)", 0.00001, 0.01, 0.0001, format="%.5f")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 4. Parâmetros Hidráulicos")
prod = st.sidebar.number_input("Produção (Mtpa)", 1.0, 100.0, 26.5)
conc = st.sidebar.slider("Conc. Sólidos (%)", 10, 80, 64)
diam = st.sidebar.selectbox("Diâmetro (pol)", [8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 36, 40], index=5)
sg_solido = st.sidebar.number_input("SG do Sólido (t/m³)", 1.5, 8.0, 4.6, help="Densidade real do minério (ex: 4.6 para hematita/itabirito)")
d50 = st.sidebar.number_input("D50 do Sólido (microns)", 10, 3000, 150, help="Diâmetro mediano das partículas (curva granulométrica)")
rugosidade_mm = st.sidebar.number_input("Rugosidade Interna do Tubo (mm)", 0.001, 1.0, 0.045, format="%.3f", help="Aço/borracha ≈0.045mm | Polietileno ≈0.015mm")
k_durand = st.sidebar.slider("Constante de Durand (K)", 100.0, 150.0, 121.0, help="Faixa típica de literatura: 121 (Durand original) a 150 (correção de Gibert)")
regra_slack_flow = st.sidebar.number_input("Regra de Slack Flow (m.c.a)", 0, 100, 20, help="Folga mínima de pressão exigida acima do terreno para evitar cavitação")
c_inicio = st.sidebar.number_input("Carga na inicio (m.c.a)", 0, 3000, 200, help="Referência real: inicios-Rio opera ~1.835 m.c.a na EB1 (bombas de deslocamento positivo)")

usar_eb2 = st.sidebar.checkbox("Forçar passagem por EB-2", value=False)
c_eb2 = 0
if usar_eb2:
    c_eb2 = st.sidebar.number_input("Carga na EB-2 (m.c.a)", 0, 3000, 200, help="Referência real: inicios-Rio opera ~2.039 m.c.a na EB2 (bombas de deslocamento positivo)")

terreno, trans, crs, bounds, pixel_size = carregar_terreno(fator, tiff_path)
if terreno is None: st.error("Erro TIFF."); st.stop()

if st.session_state['inicio_coords'] is None:
    st.session_state['inicio_coords'] = (bounds.left*0.9+bounds.right*0.1, bounds.bottom*0.1+bounds.top*0.9)
    st.session_state['fim_coords'] = (bounds.left*0.1+bounds.right*0.9, bounds.bottom*0.9+bounds.top*0.1)

st.sidebar.markdown("---")
st.sidebar.header("📍 5. Posicionamento (Coordenadas)")
modo = st.sidebar.radio("Mover Clique no Mapa:", ("Nenhum", "🟢 Inicio", "🔵 Fim", "📍 EB-2"), index=0)

c1, c2 = st.sidebar.columns(2)
mx = c1.number_input("inicio X", value=st.session_state['inicio_coords'][0], format="%.2f")
my = c2.number_input("inicio Y", value=st.session_state['inicio_coords'][1], format="%.2f")
st.session_state['inicio_coords'] = (mx, my)

c3, c4 = st.sidebar.columns(2)
px = c3.number_input("fim X", value=st.session_state['fim_coords'][0], format="%.2f")
py = c4.number_input("fim Y", value=st.session_state['fim_coords'][1], format="%.2f")
st.session_state['fim_coords'] = (px, py)

if usar_eb2:
    if st.session_state['eb2_coords'] is None: st.session_state['eb2_coords'] = ((mx+px)/2, (my+py)/2)
    c5, c6 = st.sidebar.columns(2)
    ex = c5.number_input("EB-2 X", value=st.session_state['eb2_coords'][0], format="%.2f")
    ey = c6.number_input("EB-2 Y", value=st.session_state['eb2_coords'][1], format="%.2f")
    st.session_state['eb2_coords'] = (ex, ey)

# --- 5. MOTOR DE CUSTOS E DIJKSTRA ---
tiff_bounds = get_tiff_bounds(tiff_path)

# Custo por DECLIVIDADE REAL (gradiente entre pixels vizinhos), nao por cota bruta.
# Usar a elevacao absoluta (terreno * peso) fazia o vale do rio - a regiao de menor
# cota da bacia - virar sistematicamente o caminho "mais barato", entao a rota
# tendia a seguir o curso d'agua. Com o gradiente, o custo reflete o quanto o
# terreno varia ali, nao a altitude em si.
gy, gx = np.gradient(terreno.astype(float), pixel_size * fator)
declividade = np.sqrt(gx**2 + gy**2)  # m/m
custo_base = (declividade * peso_declive) + 1.0
custo_base[terreno <= 0] += CUSTO_AREA_INVALIDA

gdfs_carregados = []
gdf_hidro_carregado = None

with rasterio.open(tiff_path) as src:
    crs_str = str(src.crs)
    transform_gdal = src.transform.to_gdal()

    if up_shapes:
        for up in up_shapes:
            custo_shp, gdf_restricao = preparar_custo_shapefile(
                up, tiff_bounds, crs_str, transform_gdal, terreno.shape, fator,
                CUSTO_RESTRICAO_SHAPEFILE
            )
            if gdf_restricao is not None:
                gdfs_carregados.append(gdf_restricao)
                custo_base += custo_shp

    if up_hidrografia:
        # buffer + simplificacao + rasterizacao acontecem dentro da funcao
        # cacheada -- so reprocessa de fato se o arquivo, o bbox ou a faixa
        # (buffer_hidrografia) mudarem, nao a cada interacao no app.
        custo_hidro, gdf_hidro_carregado = preparar_custo_shapefile(
            up_hidrografia, tiff_bounds, crs_str, transform_gdal, terreno.shape, fator,
            CUSTO_RESTRICAO_HIDROGRAFIA, buffer_m=buffer_hidrografia
        )
        if custo_hidro is not None:
            custo_base += custo_hidro

    if st.session_state.get('user_drawings'):
        val = [f for f in st.session_state['user_drawings'] if f.get('geometry')]
        if val:
            fc = {"type": "FeatureCollection", "features": val}
            gdf_draw = gpd.GeoDataFrame.from_features(fc).set_crs(epsg=4326).to_crs(src.crs)
            gdf_draw = corrigir_geometrias_invalidas(gdf_draw)
            custo_base += rasterizar_shp_simples(gdf_draw, src.crs, src.transform, terreno.shape, fator, CUSTO_RESTRICAO_DESENHO_MANUAL)

pontos_idx = []
with rasterio.open(tiff_path) as src:
    try:
        r, c = src.index(*st.session_state['inicio_coords'])
        pontos_idx.append((min(int(r/fator), terreno.shape[0]-1), min(int(c/fator), terreno.shape[1]-1)))
        if usar_eb2 and st.session_state['eb2_coords']:
            r, c = src.index(*st.session_state['eb2_coords'])
            pontos_idx.append((min(int(r/fator), terreno.shape[0]-1), min(int(c/fator), terreno.shape[1]-1)))
        r, c = src.index(*st.session_state['fim_coords'])
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
            idx, _ = route_through_array(matriz, pontos[i], pontos[i+1], geometric=True)
            if i == 0 and len(pontos) == 3: split_idx = len(idx) - 1
            if i > 0: idx = idx[1:] 
            segmentos.append(np.array(idx))
        return np.vstack(segmentos), split_idx
    except Exception as e:
        st.error(f"Erro ao calcular a rota: {e}")
        return None, None

with st.spinner("Calculando Rotas (Otimizada e Gravidade)..."):
    rota_n, split_n = calcular_segmentos(pontos_idx, custo_base)
    rota_g, split_g = calcular_segmentos(pontos_idx, custo_grav)

# --- 6. CÁLCULO HIDRÁULICO (HGL E FÍSICA) ---
D50_LIMITE_HOMOGENEO_MICRON = 40.0  # abaixo disso a particula e fina o bastante para
                                     # se manter em suspensao por turbulencia/efeitos
                                     # viscosos (nao mais o regime heterogeneo classico
                                     # de Durand-Condolios). Ex: concentrados de flotacao
                                     # (Minas-Rio: 86% passante em 44um) caem nessa faixa.

def calcular_gradiente_durand(vel, diam_m, sg_mix, sg_solido_local, d50_micron, rugosidade_m,
                               k_durand_local=121.0, sg_liquido=1.0, visc_cp=1.0):
    """
    Calcula o gradiente hidraulico da polpa (J, em m.c.a/m).

    Usa dois regimes diferentes conforme o tamanho de particula (d50):

    - HETEROGENEO (d50 >= D50_LIMITE_HOMOGENEO_MICRON): Darcy-Weisbach + Swamee-Jain
      para o fluido carreador, combinado com a correlacao classica de Durand-Condolios
      (1952) para o efeito adicional dos solidos grossos que tendem a se depositar.
      Essa correlacao SO E VALIDA para V >= VL (regime totalmente suspenso). Quando
      V < VL, o modelo esta sendo extrapolado fora do seu dominio -- o resultado de J
      e sinalizado como nao confiavel (modelo_valido=False) em vez de deixar a formula
      divergir silenciosamente para valores fisicamente absurdos.

    - HOMOGENEO / PSEUDO-HOMOGENEO (d50 < D50_LIMITE_HOMOGENEO_MICRON): concentrados
      ultrafinos (ex: minerodutos de concentrado de flotacao, tipicamente d50 na casa
      de 20-40 microns) permanecem em suspensao mesmo abaixo do VL classico de Durand --
      por isso operam normalmente em velocidades que a correlacao heterogenea rejeitaria.
      Nesse regime a perda de carga da polpa e aproximada pela perda de carga da agua
      escalada pela densidade da mistura (aproximacao classica de escoamento
      pseudo-homogeneo -- ver Wasp et al.); um modelo mais rigoroso (ex: Wilson) exigiria
      dados reologicos que este formulario nao coleta.

    Referencias: Durand & Condolios (1952); formula do Fator de Durand conforme
    amplamente citada na literatura (ex: Arabian J. Sci. Eng. 2022); metodo pratico
    Darcy-Weisbach/Swamee-Jain conforme manual Warman/Weir (AusIMM Slurry Pumping Toolbox);
    aproximacao pseudo-homogenea conforme Wasp, Kenny & Gandhi (1977).
    """
    g = 9.81
    vel = max(vel, 1e-6)  # evita divisao por zero em trechos de velocidade ~0

    # Concentracao volumetrica de solidos (%), a partir das SGs da mistura/solido/liquido
    cv_pct = np.clip((sg_mix - sg_liquido) / (sg_solido_local - sg_liquido) * 100, 0.1, 45)
    cv_frac = cv_pct / 100

    # --- Perda de carga do fluido carreador (agua) via Darcy-Weisbach + Swamee-Jain ---
    visc_m2s = (visc_cp * 1e-3) / (sg_liquido * 1000)  # viscosidade cinematica
    re = vel * diam_m / visc_m2s
    re = max(re, 4000)  # regime turbulento, limite pratico do metodo
    f_darcy = 0.25 / (np.log10((rugosidade_m / (3.7 * diam_m)) + (5.74 / re**0.9)))**2
    i_agua = f_darcy * (vel**2) / (2 * g * diam_m)  # m/m

    # --- Fator de Durand (FL) e Velocidade Limite de Deposicao (VL) ---
    # Calculados sempre (uteis como referencia/alerta), mesmo no regime homogeneo.
    fl = 0.4794 + 0.5429 * (0.01 * cv_pct)**0.1058 * (np.log10(d50_micron) - 1)
    fl = max(fl, 0.3)
    vl = fl * np.sqrt(2 * g * diam_m * (sg_solido_local - sg_liquido) / sg_liquido)

    modelo_valido = True
    aviso = None

    if d50_micron < D50_LIMITE_HOMOGENEO_MICRON:
        # --- Regime pseudo-homogeneo (particula fina, ex: concentrado de flotacao) ---
        regime = 'homogeneo'
        j_polpa = i_agua * (sg_mix / sg_liquido)
        phi = None
        aviso = (f"d50={d50_micron:.0f}um < {D50_LIMITE_HOMOGENEO_MICRON:.0f}um: tratado como "
                 f"escoamento pseudo-homogeneo (aproximacao). Para dimensionamento final, "
                 f"validar com modelo reologico (ex: Wilson) e dados de laboratorio.")
    else:
        # --- Regime heterogeneo classico (Durand-Condolios) ---
        regime = 'heterogeneo'
        fr2 = (vel**2) / (g * diam_m * (sg_solido_local / sg_liquido - 1))
        fr2_piso = 2 * fl**2  # equivale ao Fr^2 calculado em V = VL
        fr2 = max(fr2, fr2_piso)
        phi = k_durand_local * fr2**(-1.5)
        j_polpa = i_agua * (1 + cv_frac * phi)

        if vel < vl:
            modelo_valido = False
            aviso = (f"V ({vel:.2f} m/s) abaixo de VL ({vl:.2f} m/s): a correlacao de "
                     f"Durand-Condolios esta sendo extrapolada fora do seu dominio de "
                     f"validade (valida so para V >= VL). O valor de J abaixo NAO E "
                     f"CONFIAVEL -- aumente o diametro/velocidade, reduza d50, ou use um "
                     f"modelo reologico adequado para escoamento com deposicao parcial.")

    return {
        'J': j_polpa, 'FL': fl, 'VL': vl, 'Cv_pct': cv_pct, 'i_agua': i_agua, 'Phi': phi,
        'regime': regime, 'modelo_valido': modelo_valido, 'aviso': aviso
    }


@st.cache_data(show_spinner=False)
def processar_completo(rota_idx, tolerancia, split_point, carga_ini, carga_mid,
                        tiff_path, fator, terreno, conc, prod, diam, sg_solido,
                        d50, rugosidade_mm, k_durand):
    if rota_idx is None: return None
    pr, pc = rota_idx[:,0], rota_idx[:,1]
    
    with rasterio.open(tiff_path) as src:
        lats, lons = pixel_latlon(src, pr*fator, pc*fator)
        
    line = LineString(zip(lons, lats)).simplify(tolerancia, preserve_topology=True)
    if line.geom_type == 'LineString': flons, flats = line.xy
    else: flons, flats = lons, lats
        
    elev = terreno[pr, pc]
    # Distancia real (geodesica) entre pontos consecutivos, em metros.
    # Usar pixel_size*fator direto so funciona se o raster estiver numa CRS
    # projetada em metros (ex: UTM); se o TIFF estiver em graus (EPSG:4326),
    # isso subestima a distancia por um fator de ~100.000x. Calcular a
    # distancia geodesica a partir de lat/lon resolve isso independente da
    # CRS de origem do raster.
    geod = Geod(ellps="WGS84")
    _, _, dist_step = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    dist_acum = np.insert(np.cumsum(dist_step), 0, 0)
    
    diam_m = diam * 0.0254
    area = np.pi * (diam_m/2)**2
    cw = conc/100
    sg = 100 / ((conc/sg_solido) + ((100-conc)/1.0))
    vazao_m3s = ((prod*1e6)/(365*24*0.92)) / cw / sg / 3600
    vel = vazao_m3s / area
    rugosidade_m = rugosidade_mm / 1000
    durand = calcular_gradiente_durand(vel, diam_m, sg, sg_solido, d50, rugosidade_m, k_durand)
    j = durand['J']
    
    hgl = np.zeros_like(elev, dtype=float)
    perda = j * dist_acum 
    
    if split_point is not None and carga_mid > 0:
        hgl[:split_point+1] = (elev[0] + carga_ini) - perda[:split_point+1]
        perda_relativa = perda[split_point:] - perda[split_point]
        hgl[split_point:] = (elev[split_point] + carga_mid) - perda_relativa
        amt1 = max(0, (elev[split_point] - elev[0]) + perda[split_point])
        amt2 = max(0, (elev[-1] - elev[split_point]) + perda_relativa[-1])
        amt_total = amt1 + amt2
    else:
        hgl = (elev[0] + carga_ini) - perda
        desnivel = elev[-1] - elev[0]
        amt_total = max(0, desnivel + perda[-1])
        
    pot = ((sg*1000) * 9.81 * vazao_m3s * amt_total / 0.90) / 1e6
    
    return {
        'lats': list(lats), 'lons': list(lons),
        'lats_draw': list(flats), 'lons_draw': list(flons),
        'dist_arr': dist_acum/1000, 
        'elev': elev, 'hgl': hgl, 'dist_total': dist_acum[-1]/1000, 'potencia': pot,
        'v_op': vel, 'v_limite_deposicao': durand['VL'], 'fl_durand': durand['FL'],
        'cv_pct': durand['Cv_pct'], 'j_gradiente': durand['J'],
        'regime_hidraulico': durand['regime'], 'modelo_valido': durand['modelo_valido'],
        'aviso_modelo': durand['aviso']
    }

res_n = processar_completo(rota_n, tol_geo, split_n, c_inicio, c_eb2,
                            tiff_path, fator, terreno, conc, prod, diam,
                            sg_solido, d50, rugosidade_mm, k_durand)
res_g = processar_completo(rota_g, tol_geo, split_g, c_inicio, c_eb2,
                            tiff_path, fator, terreno, conc, prod, diam,
                            sg_solido, d50, rugosidade_mm, k_durand)

# --- 7. TABS DE VISUALIZAÇÃO ---
st.title("Sistema de Otimização e Viabilidade de Minerodutos")

tab1, tab2, tab3 = st.tabs(["🌍 Mapa Interativo", "⚡ Perfil Hidráulico (HGL)", "📄 Relatório e Dados (Download)"])

with tab1:
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
        
        folium.GeoJson(
            gdf_mapa, 
            style_function=lambda x: {'color': 'orange', 'fillColor': 'orange', 'fillOpacity': 0.4, 'weight': 2},
            name="Restrição Zip"
        ).add_to(m)

    if gdf_hidro_carregado is not None:
        gdf_hidro_mapa = gdf_hidro_carregado if gdf_hidro_carregado.crs == "EPSG:4326" else gdf_hidro_carregado.to_crs("EPSG:4326")
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
            
    if st.session_state['inicio_coords']: add_marker(st.session_state['inicio_coords'], 'green', 'inicio', 'play')
    if st.session_state['fim_coords']: add_marker(st.session_state['fim_coords'], 'blue', 'fim', 'stop')
    if usar_eb2 and st.session_state['eb2_coords']: add_marker(st.session_state['eb2_coords'], 'orange', 'EB-2', 'flag')

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
            if modo == "🟢 Inicio": st.session_state['inicio_coords'] = (x, y); st.rerun()
            elif modo == "🔵 Fim": st.session_state['fim_coords'] = (x, y); st.rerun()
            elif modo == "📍 EB-2": st.session_state['eb2_coords'] = (x, y); st.rerun()

with tab2:
    if res_n:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['elev'], fill='tozeroy', line=dict(color='red', width=1), name='Terreno (Otimizada)'))
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['hgl'], line=dict(color='blue', width=2), name='HGL (Otimizada)'))
        fig.add_trace(go.Scatter(x=res_n['dist_arr'], y=res_n['elev']+regra_slack_flow, line=dict(color='orange', dash='dot'), name=f'Limite de Slack Flow (+{regra_slack_flow}m)'))
        
        if res_g:
            fig.add_trace(go.Scatter(x=res_g['dist_arr'], y=res_g['elev'], line=dict(color='purple', width=1, dash='dash'), name='Terreno (Gravidade)', visible='legendonly'))
            fig.add_trace(go.Scatter(x=res_g['dist_arr'], y=res_g['hgl'], line=dict(color='lightblue', width=2, dash='dash'), name='HGL (Gravidade)', visible='legendonly'))

        fig.update_layout(title="Perfil Piezométrico e Análise Anti-Cavitação", xaxis_title="km", yaxis_title="Elevação (m.c.a)", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if not res_n.get('modelo_valido', True):
            st.warning(f"⚠️ **Atenção — modelo fora do domínio de validade:** {res_n['aviso_modelo']}")
        elif res_n.get('regime_hidraulico') == 'homogeneo':
            st.info(f"ℹ️ {res_n['aviso_modelo']}")

        c1, c2, c3 = st.columns(3)
        slack_margin = np.min(res_n['hgl'] - res_n['elev'])
        c1.metric("Velocidade Operação", f"{res_n['v_op']:.2f} m/s")
        c2.metric("Menor Margem de Pressão", f"{slack_margin:.1f} m.c.a")
        c3.metric("Vel. Limite de Depósito (VL)", f"{res_n['v_limite_deposicao']:.2f} m/s")

        if slack_margin >= regra_slack_flow: st.success(f"✅ Pressão Segura contra cavitação (> {regra_slack_flow}m de folga).")
        else: st.error(f"❌ ALERTA: Risco de Cavitação / Slack Flow ({slack_margin:.1f}m.c.a, regra: {regra_slack_flow}m). Aumente a bomba.")

        if res_n.get('regime_hidraulico') == 'homogeneo':
            st.caption("Regime pseudo-homogêneo: VL/FL de Durand são apenas referência — não se aplicam como critério direto de depósito para partículas nessa faixa de tamanho.")
        elif res_n['v_op'] >= res_n['v_limite_deposicao']:
            st.success(f"✅ Velocidade de operação acima da VL de Durand — sem risco de sedimentação/entupimento.")
        else:
            st.error(f"❌ ALERTA: V ({res_n['v_op']:.2f} m/s) abaixo da Velocidade Limite de Depósito ({res_n['v_limite_deposicao']:.2f} m/s). Risco de formação de leito de sólidos / entupimento. Reduza o diâmetro ou aumente a vazão.")

with tab3:
    st.subheader("📥 Exportação de Resultados")
    if res_n:
        c1, c2 = st.columns(2)
        
        df_rota = pd.DataFrame({
            'Distancia_km': res_n['dist_arr'],
            'Latitude': res_n['lats'],
            'Longitude': res_n['lons'],
            'Elevacao_terreno_m': res_n['elev'],
            'Pressao_HGL_m': res_n['hgl']
        })
        csv = df_rota.to_csv(index=False).encode('utf-8')
        
        with c1:
            st.markdown("#### Planilha de Coordenadas")
            st.write("Exporte o traçado detalhado (metro a metro) incluindo as cotas topográficas e o gradiente hidráulico.")
            st.download_button(
                label="📊 Baixar Rota Otimizada (.CSV)",
                data=csv,
                file_name='rota_mineroduto.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with c2:
            st.markdown("#### Memorial Descritivo")
            st.write("Gere um documento profissional detalhando as premissas, a física do transporte e as validações de segurança.")
            inputs_pdf = {'prod':prod, 'diam':diam, 'conc':conc, 'peso':peso_declive, 'eb2':usar_eb2, 'tol_geo': tol_geo, 'carga_inicio': c_inicio, 'carga_eb2': c_eb2, 'sg_solido': sg_solido, 'd50': d50, 'slack_flow': regra_slack_flow}
            
            if st.button("🖨️ Baixar Memorial de Cálculo (.PDF)", use_container_width=True):
                with st.spinner("Gerando PDF..."):
                    pdf_data = gerar_pdf_bytes(inputs_pdf, res_n, res_g, fator, tiff_path, sid)
                    st.download_button(
                        label="Clique aqui para salvar o PDF",
                        data=pdf_data,
                        file_name='Memorial_Calculo_TCC.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
                    st.success("Memorial PDF gerado com sucesso!")
        
        st.markdown("---")
        st.write("Visualização Rápida dos Dados (CSV):")
        st.dataframe(df_rota.head(10), use_container_width=True)

    else:
        st.info("Calcule uma rota nas abas anteriores primeiro para visualizar e exportar os dados.")
