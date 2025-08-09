# --- Imports ---
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import customtkinter as ctk
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywt
import time
from PIL import Image, ImageDraw, ImageFont
import textwrap

# --- Consts e Configs ---
# Cores da UI
CORES = {
    "navy": "#0E1627", "prune": "#7F6269", "mauve": "#BD8E89",
    "rosa": "#E5C5C1", "blush": "#F4E1E0", "text_dark": "#0E1627",
    "amarelo": "#FFD700"
}

# Bandas de freq. p/ análise
BANDAS = {'theta': [4, 8], 'beta': [13, 30]}
FS_AMOSTRA = 256 # Freq. amostragem em Hz

# --- Funcs de Análise de Sinal ---
def filtrar_passa_banda_sinal(dados_sinal, corte_inf_hz, corte_sup_hz, fs_hz, ordem=4):
    nyquist = 0.5 * fs_hz
    baixo_norm = corte_inf_hz / nyquist
    alto_norm = corte_sup_hz / nyquist
    b, a = butter(ordem, [baixo_norm, alto_norm], btype='band')
    return filtfilt(b, a, dados_sinal)

"""Calcula Razão Teta/Beta (TBR) via FFT."""
def calc_tbr_fft(sinal_eeg_raw, fs_hz, bandas_ref):
    sinal_filtrado = filtrar_passa_banda_sinal(sinal_eeg_raw, 0.5, 50, fs_hz)
    num_amostras = len(sinal_filtrado)
    janela = np.hanning(num_amostras)
    sinal_janelado = sinal_filtrado * janela
    yf = np.fft.fft(sinal_janelado)
    xf = np.fft.fftfreq(num_amostras, 1 / fs_hz)[:num_amostras//2]
    psd = (2.0/num_amostras * np.abs(yf[0:num_amostras//2]))**2

    potencia_bands = {}
    for nome_b, (f_inicio, f_fim) in bandas_ref.items():
        indices = np.where((xf >= f_inicio) & (xf <= f_fim))[0]
        potencia_bands[nome_b] = np.trapz(psd[indices], xf[indices]) if len(indices) > 0 else 0.0

    pot_beta = potencia_bands.get('beta', 0.0)
    pot_theta = potencia_bands.get('theta', 0.0)
    tbr = pot_theta / (pot_beta if pot_beta > 1e-12 else 1e-12)
    return tbr

"""Análise Tempo-Frequência (Wavelet)"""
def analise_cwt_sinal(sinal, fs_hz, wavelet_tipo='cmor1.5-1.0'):
    freqs_desejadas = np.linspace(1, 50, 100)
    escalas = pywt.scale2frequency(wavelet_tipo, freqs_desejadas) / (1/fs_hz)
    coefs, freqs = pywt.cwt(sinal, escalas, wavelet_tipo, 1/fs_hz)
    potencia = np.abs(coefs)**2
    tempos = np.linspace(0, len(sinal)/fs_hz, potencia.shape[1])
    return potencia, freqs, tempos


"""Calcula espectrograma do sinal"""
def calc_espectrograma(sinal, fs_hz):
    n_per_seg = fs_hz               # 1 seg de janela
    n_overlap = fs_hz // 2          # 50% de sobreposição
    freqs, tempos, Sxx = spectrogram(sinal, fs=fs_hz, window='hann',
                                     nperseg=n_per_seg, noverlap=n_overlap, scaling='density')
    return freqs, tempos, Sxx

# --- UI App (CustomTkinter) ---
"""App principal."""
class AppEEG(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Analisador EEG v1.0")
        self.geometry("1200x750")
        self.configure(fg_color=CORES["navy"])

        self.frm_main_ui = ctk.CTkFrame(self, width=1200, height=750, fg_color=CORES["rosa"], corner_radius=30)
        self.frm_main_ui.pack(expand=True, pady=20)
        self.frm_main_ui.pack_propagate(False)

        self.dict_telas = {}
        self._setup_telas_ui()
        self.mostrar_tela("Tela_Inicial")

    def _setup_telas_ui(self):
        self.dict_telas["Tela_Inicial"] = TelaInicialUI(self.frm_main_ui, ctrl=self)
        self.dict_telas["Tela_Resultados"] = TelaResultadosUI(self.frm_main_ui, ctrl=self)

    def mostrar_tela(self, nome_tela_id):
        for tela in self.dict_telas.values():
            tela.pack_forget()
        self.dict_telas[nome_tela_id].pack(fill="both", expand=True)

    def iniciar_analise_e_show_results(self):
        tela_init_ref = self.dict_telas["Tela_Inicial"]
        tela_results_ref = self.dict_telas["Tela_Resultados"]
        
        tela_init_ref.lbl_status_app.configure(text="Processando... Aguarde.", text_color=CORES["prune"])
        self.update_idletasks() # Força update da UI

        try:
            dados_lidos = tela_results_ref.carregar_mat_data(tela_init_ref.path_pasta_data)
            tela_results_ref.dados_eeg_carregados = dados_lidos

            analise_res = tela_results_ref.exec_analise_dados(dados_lidos)
            tela_results_ref.resultados_analise_obj = analise_res

            tela_results_ref.plotar_todos_os_grafs() # Desenha tudo
            self.mostrar_tela("Tela_Resultados")
            tela_init_ref.lbl_status_app.configure(text="Análise Completa!", text_color=CORES["mauve"])

        except FileNotFoundError as fnf_err:
            tela_init_ref.lbl_status_app.configure(text=f"Erro: {fnf_err}", text_color="red")
            messagebox.showerror("Erro Arquivo", f"Arquivo não encontrado:\n{fnf_err}")
            traceback.print_exc()
        except Exception as e_geral:
            tela_init_ref.lbl_status_app.configure(text=f"Falha: {e_geral}", text_color="red")
            messagebox.showerror("Erro Geral", f"Problema inesperado:\n{e_geral}")
            traceback.print_exc()


class TelaInicialUI(ctk.CTkFrame):
    def __init__(self, pai_frm, ctrl):
        super().__init__(pai_frm, fg_color="transparent")
        self.ctrl = ctrl
        self.path_pasta_data = "" # Guarda o caminho da pasta
        
        ctk.CTkLabel(self, text="🧠🦋", font=ctk.CTkFont(size=60)).pack(pady=(80, 10))
        ctk.CTkLabel(self, text="Analisador EEG", font=ctk.CTkFont(size=28, weight="bold"), text_color=CORES["text_dark"]).pack()
        ctk.CTkLabel(self, text="Análise de TDAH", font=ctk.CTkFont(size=18), text_color=CORES["prune"]).pack(pady=(0, 60))
        
        self.btn_sel_pasta = ctk.CTkButton(self, text="Selecionar Pasta Dados", corner_radius=15,
                                           command=self.acao_selecionar_pasta,
                                           fg_color=CORES["mauve"], text_color=CORES["blush"], hover_color=CORES["prune"])
        self.btn_sel_pasta.pack(pady=20, padx=40, ipady=10, fill="x")
        
        self.lbl_path_pasta = ctk.CTkLabel(self, text="", text_color=CORES["prune"])
        self.lbl_path_pasta.pack(pady=5)
        
        self.btn_iniciar_analise = ctk.CTkButton(self, text="INICIAR ANÁLISE", corner_radius=15,
                                                 command=ctrl.iniciar_analise_e_show_results, state="disabled",
                                                 fg_color=CORES["prune"], text_color=CORES["blush"], hover_color=CORES["mauve"],
                                                 font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_iniciar_analise.pack(pady=20, padx=40, ipady=10, fill="x")
        
        self.lbl_status_app = ctk.CTkLabel(self, text="Escolha a pasta dos dados", text_color=CORES["mauve"])
        self.lbl_status_app.pack(pady=20)
        
    def acao_selecionar_pasta(self):
        caminho = filedialog.askdirectory()
        if caminho:
            self.path_pasta_data = caminho
            self.lbl_path_pasta.configure(text=f"Pasta: ...{os.path.basename(self.path_pasta_data)}")
            self.btn_iniciar_analise.configure(state="normal")
            self.lbl_status_app.configure(text="Pronto pra rodar!", text_color=CORES["mauve"])
        else:
            self.btn_iniciar_analise.configure(state="disabled")
            self.lbl_status_app.configure(text="Nenhuma pasta selecionada.", text_color="orange")


class TelaResultadosUI(ctk.CTkFrame):
    def __init__(self, pai_frm, ctrl):
        super().__init__(pai_frm, fg_color="transparent")
        self.ctrl = ctrl
        self.dados_eeg_carregados = {}
        self.resultados_analise_obj = {}

        # Agora armazenamos dicionários: {'fig': fig, 'label': '...', 'desc': '...'}
        self.lista_de_figs = []

        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scrollable_frame.pack(fill="both", expand=True)

        header_frm = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        header_frm.pack(fill="x", padx=10, pady=10)
        btn_voltar = ctk.CTkButton(header_frm, text="← Voltar", command=lambda: ctrl.mostrar_tela("Tela_Inicial"),
                                   fg_color="transparent", text_color=CORES["prune"], hover_color=CORES["rosa"], width=50)
        btn_voltar.pack(side="left")
        ctk.CTkLabel(header_frm, text="Resultados", font=ctk.CTkFont(size=20, weight="bold"), text_color=CORES["text_dark"]).pack(side="left", expand=True)

        btn_salvar_graficos = ctk.CTkButton(header_frm, text="💾 Salvar Gráficos", command=self.salvar_todos_graficos,
                                          fg_color=CORES["mauve"], text_color=CORES["blush"], hover_color=CORES["prune"])
        btn_salvar_graficos.pack(side="right", padx=10)

        self.tab_view_res = ctk.CTkTabview(self.scrollable_frame, fg_color=CORES["rosa"])
        self.tab_view_res.pack(pady=10, padx=10, fill="both", expand=True)
        self.tab_grupo_comp = self.tab_view_res.add("Comp. Grupos")
        self.tab_indiv_analise = self.tab_view_res.add("Caso Individual")
        self.tab_medias_analise = self.tab_view_res.add("Análises Médias")
        self.tab_sliding_analise = self.tab_view_res.add("Média e Variância")
        
        self.var_sliding_adhd_tbr = ctk.StringVar(value="TBR Mais Alto")
        self.var_sliding_ctrl_tbr = ctk.StringVar(value="TBR Mais Baixo")
        self.var_sliding_gen_sel = ctk.StringVar(value="Geral")

        self.frame_plots_media_tab = None 
        self.frame_plots_sliding_tab = None 

    # ---------- utilitários internos ----------
    def _registrar_fig(self, fig, label, desc):
        """Armazena figura com rótulo (para nome do arquivo) e descrição (para legenda no PNG)."""
        self.lista_de_figs.append({'fig': fig, 'label': label, 'desc': desc})

    def _wrap_text(self, draw, text, font, max_width_px):
        """Quebra o texto em linhas para caber na largura."""
        # tenta estimativa por caracteres; ajusta empiricamente
        avg_char_w = font.getlength("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") / 52.0
        max_chars = max(int(max_width_px / max(1, avg_char_w)) - 2, 10)
        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width=max_chars))
        return lines

    def _gravar_legenda_em_png(self, caminho_png, texto, pad=16):
        """Abre o PNG, adiciona uma faixa inferior com texto e salva de volta."""
        try:
            img = Image.open(caminho_png).convert("RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            # define bloco de legenda (altura ajustada ao texto)
            max_width = img.width - pad*2
            lines = self._wrap_text(draw, texto, font, max_width)
            line_h = int(font.getbbox("Ag")[3] * 1.35)
            legenda_h = pad + len(lines)*line_h + pad

            # cria nova imagem com espaço extra
            nova = Image.new("RGB", (img.width, img.height + legenda_h), (255,255,255))
            nova.paste(img, (0,0))

            # desenha faixa
            draw2 = ImageDraw.Draw(nova)
            # fundo levemente rosado pra casar com a UI
            draw2.rectangle([0, img.height, img.width, img.height + legenda_h], fill=(245, 234, 233))
            y = img.height + pad
            for ln in lines:
                draw2.text((pad, y), ln, fill=(20,20,20), font=font)
                y += line_h

            nova.save(caminho_png, format="PNG")
        except Exception as e:
            # Se der ruim, não quebrar o fluxo de salvamento
            print(f"Falha ao gravar legenda em {caminho_png}: {e}")

    # -----------------------------------------

    def carregar_mat_data(self, dir_path):
        data = {'F-TDAH': [], 'F-Ctrl': [], 'M-TDAH': [], 'M-Ctrl': []}
        arqs_map = {'F-TDAH': 'FADHD.mat', 'F-Ctrl': 'FC.mat', 'M-TDAH': 'MADHD.mat', 'M-Ctrl': 'MC.mat'}
        
        for grupo, nome_arq in arqs_map.items():
            path_completo = os.path.join(dir_path, nome_arq)
            if not os.path.exists(path_completo):
                raise FileNotFoundError(f"Arquivo '{nome_arq}' faltando.")
            
            mat = sio.loadmat(path_completo)
            chave = next(k for k in mat if not k.startswith('__')) 
            sujeitos_raw = mat[chave].flatten() 

            for subj_data_raw in sujeitos_raw:
                subj_data = subj_data_raw[0] if isinstance(subj_data_raw, np.ndarray) and subj_data_raw.size > 0 else subj_data_raw
                if subj_data.ndim == 1: subj_data = subj_data.reshape(1, -1)
                if subj_data.shape[0] > subj_data.shape[1]: subj_data = subj_data.T 
                data[grupo].append(subj_data)
        return data

    def exec_analise_dados(self, data_carregada):
        results = {'tbr_por_grupo': {}}
        for grupo, sujeitos_sinais in data_carregada.items():
            results['tbr_por_grupo'][grupo] = [calc_tbr_fft(s[0, :], FS_AMOSTRA, BANDAS) for s in sujeitos_sinais]
        return results

    # --- MÉTODOS AUXILIARES: ESTILO E GERAR FIGURAS ---
    def _estilo_grafico_padrao(self, ax_obj, fig_obj, titulo_str):
        fig_obj.patch.set_facecolor('white')
        ax_obj.set_facecolor('white')
        ax_obj.set_title(titulo_str, color=CORES["text_dark"], fontsize=15, weight="bold")
    
        ax_obj.tick_params(axis='both', colors=CORES["text_dark"],labelsize=10)
    
        ax_obj.xaxis.label.set_color('black')
        ax_obj.yaxis.label.set_color('black')
        for spine in ax_obj.spines.values():
            spine.set_edgecolor('black')
        ax_obj.grid(True, linestyle='--', color='gray', alpha=0.5) 
        return ax_obj

    def _plot_comp_grupos_tbr_fig(self, figsize=(5, 5), fontsize_adj=0):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        tbr_data = self.resultados_analise_obj['tbr_por_grupo']
        data_plot = [tbr_data['F-TDAH'], tbr_data['F-Ctrl'], tbr_data['M-TDAH'], tbr_data['M-Ctrl']]
        labels = ['TDAH (F)', 'Controle (F)', 'TDAH (M)', 'Controle (M)']
        
        box = ax.boxplot(data_plot, labels=labels, patch_artist=True, widths=0.5)
        cores_box = [CORES["mauve"], CORES["prune"], CORES["mauve"], CORES["prune"]]
        for patch, color in zip(box['boxes'], cores_box): patch.set_facecolor(color)
        for median in box['medians']: median.set(color=CORES['amarelo'], linewidth=2)
        
        self._estilo_grafico_padrao(ax, fig, "TBR por Grupo")
        ax.set_ylabel("TBR")
        
        leg_txt = ('LEGENDA:\n' '● Amarela: Mediana\n' '■ Caixa: 50% dos dados\n' '─ Linhas: Min/Max\n' '○ Círculos: Outliers')
        props = dict(boxstyle='round,pad=0.5', facecolor=CORES["blush"], alpha=0.95, edgecolor=CORES["prune"])
        ax.text(0.03, 0.97, leg_txt, transform=ax.transAxes, fontsize=9+fontsize_adj, verticalalignment='top', horizontalalignment='left', bbox=props, color=CORES["text_dark"])
        fig.tight_layout()
        return fig, ax

    def _plot_indiv_figs(self, sinal_adhd, sinal_ctrl, figsize=(10, 3.5), fontsize_adj=0):
        # Gera a figura do Wavelet
        fig_wav, (ax_wav_adhd, ax_wav_ctrl) = plt.subplots(1, 2, figsize=figsize)

        pow_a, freqs_a, temps_a = analise_cwt_sinal(sinal_adhd, FS_AMOSTRA)
        im_a = ax_wav_adhd.contourf(temps_a, freqs_a, pow_a, levels=20, cmap='viridis')
        self._estilo_grafico_padrao(ax_wav_adhd, fig_wav, f"Wavelet TDAH ({self.var_tbr_adhd_indiv.get()})")
        barra_wav_a = fig_wav.colorbar(im_a, ax=ax_wav_adhd)
        ax_wav_adhd.set_xlabel("Tempo (s)", fontsize=12)
        ax_wav_adhd.set_ylabel("Frequência (Hz)", fontsize=12)
        barra_wav_a.ax.tick_params(labelsize=10)

        pow_c, freqs_c, temps_c = analise_cwt_sinal(sinal_ctrl, FS_AMOSTRA)
        im_c = ax_wav_ctrl.contourf(temps_c, freqs_c, pow_c, levels=20, cmap='viridis')
        self._estilo_grafico_padrao(ax_wav_ctrl, fig_wav, f"Wavelet Controle ({self.var_tbr_ctrl_indiv.get()})")
        barra_wav_c = fig_wav.colorbar(im_c, ax=ax_wav_ctrl)
        fig_wav.tight_layout()
        ax_wav_ctrl.set_xlabel("Tempo (s)", fontsize=12)
        ax_wav_ctrl.set_ylabel("Frequência (Hz)", fontsize=12)
        barra_wav_c.ax.tick_params(labelsize=10)

        # Gera a figura do Espectrograma
        fig_spec, (ax_spec_adhd, ax_spec_ctrl) = plt.subplots(1, 2, figsize=figsize)
        
        freqs_sa, temps_sa, Sxx_a = calc_espectrograma(sinal_adhd, FS_AMOSTRA)
        im_sa = ax_spec_adhd.pcolormesh(temps_sa, freqs_sa, 10*np.log10(Sxx_a), shading='gouraud', cmap='viridis')
        self._estilo_grafico_padrao(ax_spec_adhd, fig_spec, f"Espectrograma TDAH ({self.var_tbr_adhd_indiv.get()})")
        barra_espectograma_a = fig_spec.colorbar(im_sa, ax=ax_spec_adhd)
        ax_spec_adhd.set_xlabel("Tempo (s)", fontsize=12)
        ax_spec_adhd.set_ylabel("Frequência (Hz)", fontsize=12)
        barra_espectograma_a.ax.tick_params(labelsize=10)
        
        freqs_sc, temps_sc, Sxx_c = calc_espectrograma(sinal_ctrl, FS_AMOSTRA)
        im_sc = ax_spec_ctrl.pcolormesh(temps_sc, freqs_sc, 10*np.log10(Sxx_c), shading='gouraud', cmap='viridis')
        self._estilo_grafico_padrao(ax_spec_ctrl, fig_spec, f"Espectrograma Controle ({self.var_tbr_ctrl_indiv.get()})")
        barra_espectograma_c = fig_spec.colorbar(im_sc, ax=ax_spec_ctrl)
        fig_spec.tight_layout()
        ax_spec_ctrl.set_xlabel("Tempo (s)", fontsize=12)
        ax_spec_ctrl.set_ylabel("Frequência (Hz)", fontsize=12)
        barra_espectograma_c.ax.tick_params(labelsize=10)

        return fig_wav, fig_spec

    def _plot_media_espectro_fig(self, modo_gen, figsize=(10, 3.5), fontsize_adj=0):
        fig_spec, (ax_spec_adhd, ax_spec_ctrl) = plt.subplots(1, 2, figsize=figsize)
        sinais_adhd, sinais_ctrl = [], []

        if modo_gen == "Geral":
            sinais_adhd = self.dados_eeg_carregados.get('F-TDAH', []) + self.dados_eeg_carregados.get('M-TDAH', [])
            sinais_ctrl = self.dados_eeg_carregados.get('F-Ctrl', []) + self.dados_eeg_carregados.get('M-Ctrl', [])
        elif modo_gen == "Feminino":
            sinais_adhd = self.dados_eeg_carregados.get('F-TDAH', [])
            sinais_ctrl = self.dados_eeg_carregados.get('F-Ctrl', [])
        else: # Masculino
            sinais_adhd = self.dados_eeg_carregados.get('M-TDAH', [])
            sinais_ctrl = self.dados_eeg_carregados.get('M-Ctrl', [])

        if sinais_adhd:
            min_len_adhd = min(s.shape[1] for s in sinais_adhd)
            all_sxx_adhd = [calc_espectrograma(s[0, :min_len_adhd], FS_AMOSTRA)[2] for s in sinais_adhd]
            avg_sxx_adhd = np.mean(all_sxx_adhd, axis=0)
            
            freqs_adhd, temps_adhd, _ = calc_espectrograma(sinais_adhd[0][0, :min_len_adhd], FS_AMOSTRA)
            
            im_adhd = ax_spec_adhd.pcolormesh(temps_adhd, freqs_adhd, 10 * np.log10(avg_sxx_adhd), shading='gouraud', cmap='viridis')
            self._estilo_grafico_padrao(ax_spec_adhd, fig_spec, f"Espectro Médio TDAH ({modo_gen})")
            ax_spec_adhd.set_xlabel("Tempo (s)", fontsize=12)
            ax_spec_adhd.set_ylabel("Frequência (Hz)", fontsize=12)
            fig_spec.colorbar(im_adhd, ax=ax_spec_adhd).ax.tick_params(labelsize=10)
        else:
            self._estilo_grafico_padrao(ax_spec_adhd, fig_spec, f"Espectro Médio TDAH ({modo_gen})")
            ax_spec_adhd.text(0.5, 0.5, "Sem dados.", horizontalalignment='center', verticalalignment='center', transform=ax_spec_adhd.transAxes, color=CORES["text_dark"])
        
        if sinais_ctrl:
            min_len_ctrl = min(s.shape[1] for s in sinais_ctrl)
            all_sxx_ctrl = [calc_espectrograma(s[0, :min_len_ctrl], FS_AMOSTRA)[2] for s in sinais_ctrl]
            avg_sxx_ctrl = np.mean(all_sxx_ctrl, axis=0)
            
            freqs_ctrl, temps_ctrl, _ = calc_espectrograma(sinais_ctrl[0][0, :min_len_ctrl], FS_AMOSTRA)
            
            im_ctrl = ax_spec_ctrl.pcolormesh(temps_ctrl, freqs_ctrl, 10 * np.log10(avg_sxx_ctrl), shading='gouraud', cmap='viridis')
            self._estilo_grafico_padrao(ax_spec_ctrl, fig_spec, f"Espectro Médio Ctrl ({modo_gen})")
            ax_spec_ctrl.set_xlabel("Tempo (s)", fontsize=12)
            ax_spec_ctrl.set_ylabel("Frequência (Hz)", fontsize=12)
            fig_spec.colorbar(im_ctrl, ax=ax_spec_ctrl).ax.tick_params(labelsize=10)
        else:
            self._estilo_grafico_padrao(ax_spec_ctrl, fig_spec, f"Espectro Médio Ctrl ({modo_gen})")
            ax_spec_ctrl.text(0.5, 0.5, "Sem dados.", horizontalalignment='center', verticalalignment='center', transform=ax_spec_ctrl.transAxes, color=CORES["text_dark"])

        fig_spec.tight_layout()
        return fig_spec, (ax_spec_adhd, ax_spec_ctrl)

    def _plot_sliding_mean_var_fig(self, modo_gen, modo_tbr_adhd, modo_tbr_ctrl, figsize=(14, 6), fontsize_adj=0):
        sinais_adhd_filtr, tbr_adhd_filtr = [], []
        sinais_ctrl_filtr, tbr_ctrl_filtr = [], []

        if modo_gen == "Geral":
            sinais_adhd_filtr = self.dados_eeg_carregados.get('F-TDAH', []) + self.dados_eeg_carregados.get('M-TDAH', [])
            tbr_adhd_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('F-TDAH', []) + self.resultados_analise_obj['tbr_por_grupo'].get('M-TDAH', [])
            sinais_ctrl_filtr = self.dados_eeg_carregados.get('F-Ctrl', []) + self.dados_eeg_carregados.get('M-Ctrl', [])
            tbr_ctrl_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('F-Ctrl', []) + self.resultados_analise_obj['tbr_por_grupo'].get('M-Ctrl', [])
        elif modo_gen == "Feminino":
            sinais_adhd_filtr = self.dados_eeg_carregados.get('F-TDAH', [])
            tbr_adhd_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('F-TDAH', [])
            sinais_ctrl_filtr = self.dados_eeg_carregados.get('F-Ctrl', [])
            tbr_ctrl_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('F-Ctrl', [])
        else: # Masculino
            sinais_adhd_filtr = self.dados_eeg_carregados.get('M-TDAH', [])
            tbr_adhd_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('M-TDAH', [])
            sinais_ctrl_filtr = self.dados_eeg_carregados.get('M-Ctrl', [])
            tbr_ctrl_filtr = self.resultados_analise_obj['tbr_por_grupo'].get('M-Ctrl', [])
        
        if not sinais_adhd_filtr or not sinais_ctrl_filtr:
            return None, None # Retorna None se não houver dados
        
        idx_adhd = np.argmax(tbr_adhd_filtr) if modo_tbr_adhd == "TBR Mais Alto" else np.argmin(tbr_adhd_filtr)
        idx_ctrl = np.argmax(tbr_ctrl_filtr) if modo_tbr_ctrl == "TBR Mais Alto" else np.argmin(tbr_ctrl_filtr)

        sinal_adhd_sel = sinais_adhd_filtr[idx_adhd][0, :]
        sinal_ctrl_sel = sinais_ctrl_filtr[idx_ctrl][0, :]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98, hspace=0.4, wspace=0.3)

        self._plot_sliding_mean_var_aux(axes[:, 0], sinal_adhd_sel, FS_AMOSTRA, f"TDAH ({modo_gen} - {modo_tbr_adhd})", fontsize_adj)
        self._plot_sliding_mean_var_aux(axes[:, 1], sinal_ctrl_sel, FS_AMOSTRA, f"Ctrl ({modo_gen} - {modo_tbr_ctrl})", fontsize_adj)

        return fig, axes

    def _plot_sliding_mean_var_aux(self, eixos, sinal, fs_hz, prefixo_titulo, fontsize_adj=0):
        win_size_sec = 1.0 # Janela de 1s
        win_samples = int(win_size_sec * fs_hz)
        step_samples = win_samples // 4 # 25% passo
        
        means, vars, time_pts = [], [], []

        for i in range(0, len(sinal) - win_samples + 1, step_samples):
            window = sinal[i:i + win_samples]
            means.append(np.mean(window))
            vars.append(np.var(window))
            time_pts.append((i + win_samples / 2) / fs_hz)
        
        ax_mean, ax_var = eixos[0], eixos[1]
        
        ax_mean.plot(time_pts, means, color=CORES["prune"])
        self._estilo_grafico_padrao(ax_mean, ax_mean.get_figure(), f"Média - {prefixo_titulo}")
        ax_mean.set_ylabel("Média (µV)", fontsize=12)
        ax_mean.tick_params(axis='both', colors=CORES["text_dark"], labelsize=10)

        ax_var.plot(time_pts, vars, color=CORES["mauve"])
        self._estilo_grafico_padrao(ax_var, ax_var.get_figure(), f"Variância - {prefixo_titulo}")
        ax_var.set_ylabel("Variância (µV²)", fontsize=12)
        ax_var.set_xlabel("Tempo (s)", fontsize=12)
        ax_var.tick_params(axis='both', colors=CORES["text_dark"], labelsize=10)

    # --- MÉTODOS DE PLOTAGEM QUE UTILIZAM AS FIGURAS GERADAS ---
    def plotar_todos_os_grafs(self):
        # Limpa a lista de figuras antes de gerar novos gráficos
        self.lista_de_figs = []

        # Limpar os frames existentes antes de plotar
        for widget in self.tab_grupo_comp.winfo_children(): widget.destroy()
        for widget in self.tab_indiv_analise.winfo_children(): widget.destroy()
        for widget in self.tab_medias_analise.winfo_children(): widget.destroy()
        for widget in self.tab_sliding_analise.winfo_children(): widget.destroy()

        self._plot_comp_grupos_tbr_ui()
        self._setup_analise_indiv_ui()
        self._setup_analise_media_ui()
        self._setup_analise_sliding_ui()

    def _mostrar_fig_no_frame(self, fig, frame):
        """Função utilitária para exibir uma figura em um frame."""
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        plt.close(fig) # Fecha a figura para não exibir duas vezes

    def _plot_comp_grupos_tbr_ui(self):
        fig, _ = self._plot_comp_grupos_tbr_fig()
        self._mostrar_fig_no_frame(fig, self.tab_grupo_comp)
        # registrar com rótulo e descrição
        desc = ("Boxplot da Razão Teta/Beta (TBR) por grupo: TDAH (F/M) e Controle (F/M). "
                "Bandas: θ 4–8 Hz, β 13–30 Hz; fs=256 Hz. "
                "TBR = potência θ / β; mediana em amarelo; caixas representam o IQR.")
        self._registrar_fig(fig, "Grupos", desc)
        
        ctk.CTkButton(self.tab_grupo_comp, text="💡 Entender TBR", corner_radius=10, fg_color=CORES["mauve"], text_color=CORES["blush"],
                      command=lambda: messagebox.showinfo("TBR Insight", "Uma Razão Teta/Beta (TBR) mais alta no grupo TDAH é um biomarcador comum, sugerindo uma maturação cerebral mais lenta. A separação por gênero permite investigar se há diferenças neste padrão.")).pack(pady=10, padx=20)
    
    def _setup_analise_indiv_ui(self):
        for widget in self.tab_indiv_analise.winfo_children(): widget.destroy()
        ctrl_pnl = ctk.CTkFrame(self.tab_indiv_analise, fg_color=CORES["blush"])
        ctrl_pnl.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(ctrl_pnl, text="Caso TDAH:", text_color=CORES["text_dark"]).pack(side="left", padx=(10,5))
        self.var_tbr_adhd_indiv = ctk.StringVar(value="TBR Mais Alto")
        ctk.CTkComboBox(ctrl_pnl, variable=self.var_tbr_adhd_indiv, values=["TBR Mais Alto", "TBR Mais Baixo"]).pack(side="left", padx=5)
        
        ctk.CTkLabel(ctrl_pnl, text="Caso Ctrl:", text_color=CORES["text_dark"]).pack(side="left", padx=(20,5))
        self.var_tbr_ctrl_indiv = ctk.StringVar(value="TBR Mais Baixo")
        ctk.CTkComboBox(ctrl_pnl, variable=self.var_tbr_ctrl_indiv, values=["TBR Mais Alto", "TBR Mais Baixo"]).pack(side="left", padx=5)
        
        ctk.CTkButton(ctrl_pnl, text="Atualizar", command=self.atualizar_plots_indiv).pack(side="right", padx=10)
        
        self.frm_wav_plot = ctk.CTkFrame(self.tab_indiv_analise, fg_color="transparent")
        self.frm_wav_plot.pack(fill="both", expand=True, pady=5)
        self.frm_spec_plot = ctk.CTkFrame(self.tab_indiv_analise, fg_color="transparent")
        self.frm_spec_plot.pack(fill="both", expand=True, pady=5)
        self.atualizar_plots_indiv()

    def atualizar_plots_indiv(self):
        all_adhd_tbr = self.resultados_analise_obj['tbr_por_grupo'].get('F-TDAH', []) + self.resultados_analise_obj['tbr_por_grupo'].get('M-TDAH', [])
        all_adhd_sinais = self.dados_eeg_carregados.get('F-TDAH', []) + self.dados_eeg_carregados.get('M-TDAH', [])
        all_ctrl_tbr = self.resultados_analise_obj['tbr_por_grupo'].get('F-Ctrl', []) + self.resultados_analise_obj['tbr_por_grupo'].get('M-Ctrl', [])
        all_ctrl_sinais = self.dados_eeg_carregados.get('F-Ctrl', []) + self.dados_eeg_carregados.get('M-Ctrl', [])
        
        if not all_adhd_sinais or not all_ctrl_sinais: return # Sem dados, sai

        idx_adhd = np.argmax(all_adhd_tbr) if self.var_tbr_adhd_indiv.get() == "TBR Mais Alto" else np.argmin(all_adhd_tbr)
        idx_ctrl = np.argmax(all_ctrl_tbr) if self.var_tbr_ctrl_indiv.get() == "TBR Mais Alto" else np.argmin(all_ctrl_tbr)
        
        sinal_adhd = all_adhd_sinais[idx_adhd][0, :]
        sinal_ctrl = all_ctrl_sinais[idx_ctrl][0, :]
        
        fig_wav, fig_spec = self._plot_indiv_figs(sinal_adhd, sinal_ctrl)
        
        # Registrar com rótulos e descrições
        desc_wav = (f"Análise Wavelet Contínua (CWT, cmor1.5-1.0) — TDAH ({self.var_tbr_adhd_indiv.get()}) vs "
                    f"Controle ({self.var_tbr_ctrl_indiv.get()}). Potência |coef|^2; frequências 1–50 Hz; fs=256 Hz.")
        desc_spec = (f"Espectrograma (janela Hann 1s, 50% overlap) — TDAH ({self.var_tbr_adhd_indiv.get()}) vs "
                     f"Controle ({self.var_tbr_ctrl_indiv.get()}). Escala em dB; fs=256 Hz.")
        self._registrar_fig(fig_wav, "Indiv_Wavelet", desc_wav)
        self._registrar_fig(fig_spec, "Indiv_Espectrograma", desc_spec)
        
        self._mostrar_fig_no_frame(fig_wav, self.frm_wav_plot)
        self._mostrar_fig_no_frame(fig_spec, self.frm_spec_plot)

    def _setup_analise_media_ui(self):
        for widget in self.tab_medias_analise.winfo_children(): widget.destroy()
        self.genero_selecionar_media = ctk.CTkSegmentedButton(self.tab_medias_analise, values=["Geral", "Feminino", "Masculino"],
                                                              command=self.atualizar_plots_media, fg_color=CORES["mauve"],
                                                              selected_color=CORES["prune"], selected_hover_color=CORES["prune"],
                                                              unselected_color=CORES["mauve"], unselected_hover_color=CORES["rosa"],
                                                              text_color=CORES["blush"])
        self.genero_selecionar_media.set("Geral")
        self.genero_selecionar_media.pack(pady=10)
        self.frame_plots_media_tab = ctk.CTkFrame(self.tab_medias_analise, fg_color="transparent")
        self.frame_plots_media_tab.pack(fill="both", expand=True)
        self.atualizar_plots_media("Geral")
        
    def atualizar_plots_media(self, modo_gen):
        fig_spec, _ = self._plot_media_espectro_fig(modo_gen)
        if fig_spec:
            self._mostrar_fig_no_frame(fig_spec, self.frame_plots_media_tab)
            desc = (f"Espectrograma médio por grupo ({modo_gen}). Média dos Sxx por sujeito; "
                    "janela Hann 1s, 50% overlap; escala em dB; fs=256 Hz.")
            self._registrar_fig(fig_spec, f"EspectroMedio_{modo_gen}", desc)

    def _setup_analise_sliding_ui(self):
        for widget in self.tab_sliding_analise.winfo_children(): widget.destroy()

        ctrl_pnl = ctk.CTkFrame(self.tab_sliding_analise, fg_color=CORES["blush"])
        ctrl_pnl.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(ctrl_pnl, text="Caso TDAH:", text_color=CORES["text_dark"]).pack(side="left", padx=(15,10))
        ctk.CTkComboBox(ctrl_pnl, variable=self.var_sliding_adhd_tbr, values=["TBR Mais Alto", "TBR Mais Baixo"],
                                 command=lambda x: self.atualizar_plots_sliding()).pack(side="left", padx=5)

        ctk.CTkLabel(ctrl_pnl, text="Caso Controle:", text_color=CORES["text_dark"]).pack(side="left", padx=(15,10))
        ctk.CTkComboBox(ctrl_pnl, variable=self.var_sliding_ctrl_tbr, values=["TBR Mais Alto", "TBR Mais Baixo"],
                                 command=lambda x: self.atualizar_plots_sliding()).pack(side="left", padx=5)
        
        self.sel_genero_sliding = ctk.CTkSegmentedButton(ctrl_pnl, values=["Geral", "Feminino", "Masculino"],
                                                         command=self.atualizar_plots_sliding, fg_color=CORES["mauve"],
                                                         selected_color=CORES["prune"], selected_hover_color=CORES["prune"],
                                                         unselected_color=CORES["mauve"], unselected_hover_color=CORES["rosa"],
                                                         text_color=CORES["blush"], variable=self.var_sliding_gen_sel)
        self.sel_genero_sliding.set("Geral")
        self.sel_genero_sliding.pack(side="right", padx=10)
        
        self.frame_plots_sliding_tab = ctk.CTkFrame(self.tab_sliding_analise, fg_color="transparent")
        self.frame_plots_sliding_tab.pack(fill="both", expand=True, pady=5)
        self.atualizar_plots_sliding()

    def atualizar_plots_sliding(self, *args):
        modo_gen = self.var_sliding_gen_sel.get()
        modo_tbr_adhd = self.var_sliding_adhd_tbr.get()
        modo_tbr_ctrl = self.var_sliding_ctrl_tbr.get()

        fig, _ = self._plot_sliding_mean_var_fig(modo_gen, modo_tbr_adhd, modo_tbr_ctrl)
        if fig:
            self._mostrar_fig_no_frame(fig, self.frame_plots_sliding_tab)
            desc = (f"Média e Variância deslizantes — {modo_gen}; seleção: "
                    f"TDAH({modo_tbr_adhd}) vs Ctrl({modo_tbr_ctrl}). Janela 1s, passo 25%, fs=256 Hz.")
            self._registrar_fig(fig, f"MediaVar_{modo_gen}", desc)
        else:
            for widget in self.frame_plots_sliding_tab.winfo_children(): widget.destroy()
            ctk.CTkLabel(self.frame_plots_sliding_tab, text="Sem dados para esta seleção.",
                         text_color=CORES["text_dark"], font=ctk.CTkFont(size=14)).pack(pady=50)

    # --- SALVAR GRÁFICOS ---
    def salvar_todos_graficos(self):
        """Salva todos os gráficos gerados com nome numerado e rótulo, e adiciona descrição no PNG."""
        caminho_pasta = filedialog.askdirectory(title="Selecione a pasta para salvar os gráficos")
        if not caminho_pasta:
            messagebox.showinfo("Cancelado", "Operação de salvar cancelada.")
            return

        try:
            if not self.lista_de_figs:
                messagebox.showerror("Erro ao Salvar", "Nenhum gráfico gerado para salvar.")
                return

            for i, item in enumerate(self.lista_de_figs, start=1):
                fig = item['fig']
                label = item.get('label', f"Grafico{i:02d}")
                desc = item.get('desc', '')

                nome_arquivo = f"{i:02d}_{label}.png"
                path_completo = os.path.join(caminho_pasta, nome_arquivo)

                # Salva figura em alta qualidade
                fig.savefig(path_completo, dpi=300)
                # Acrescenta legenda com descrição
                if desc:
                    self._gravar_legenda_em_png(path_completo, desc)

            messagebox.showinfo("Sucesso", f"Todos os gráficos foram salvos na pasta:\n{caminho_pasta}")

        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Ocorreu um erro ao salvar os gráficos:\n{e}")
            traceback.print_exc()

# --- Bloco Exec Principal ---
if __name__ == "__main__":
    app = AppEEG()
    app.mainloop()
