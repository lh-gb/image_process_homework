"""Tkinter 桌面演示；输入与结果均在本地处理。"""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageOps, ImageDraw
from docscan.io import load_image
from docscan.config import DetectionConfig, EnhancementConfig
from docscan.pipeline import process_document, export_result

ROOT=Path(__file__).resolve().parent
ENHANCEMENTS={'光照补偿 + 局部阈值':'illumination','局部自适应阈值':'adaptive','全局 Otsu 阈值':'otsu'}
DETECTORS={'多通道候选与评分':'robust','固定边缘基线':'baseline'}
VIEWS={'黑白扫描':'binary','灰度增强':'enhanced','展正彩色':'rectified'}

class DocumentScannerApp:
    def __init__(self,root):
        self.root=root; root.title('手机文档 · 透视校正与增强'); root.geometry('1260x830'); root.minsize(960,650)
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.image=None; self.result=None; self.path=None
        self.points=[]; self.manual=False; self.busy=False; self._photos={}; self._transform=None
        style=ttk.Style(root); style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f3f5f9',foreground='#24334b')
        style.configure('TButton',padding=(12,8)); style.configure('TCombobox',padding=5)
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',21,'bold'),foreground='#152742')
        style.configure('Accent.TButton',background='#245cbb',foreground='white')
        style.map('Accent.TButton',background=[('active','#194791')])
        root.configure(bg='#f3f5f9')
        page=ttk.Frame(root,padding=22); page.pack(fill='both',expand=True)
        ttk.Label(page,text='手机拍摄文档透视校正与增强',style='Title.TLabel').pack(anchor='w')
        ttk.Label(page,text='选择照片，检查四角，生成展正与扫描结果。',foreground='#68778c').pack(anchor='w',pady=(5,18))
        row=ttk.Frame(page); row.pack(fill='x')
        ttk.Button(row,text='打开照片',command=self.open_file).pack(side='left')
        self.files=sorted((ROOT/'dataset').glob('doc_*.jpg'))
        self.file_var=tk.StringVar()
        combo=ttk.Combobox(row,textvariable=self.file_var,values=[p.name for p in self.files],state='readonly',width=19)
        combo.pack(side='left',padx=(10,20)); combo.bind('<<ComboboxSelected>>',self.select_dataset)
        self.detector=tk.StringVar(value=next(iter(DETECTORS)))
        ttk.Label(row,text='定位').pack(side='left',padx=(0,6))
        ttk.Combobox(row,textvariable=self.detector,values=list(DETECTORS),state='readonly',width=19).pack(side='left')
        self.enhancement=tk.StringVar(value=next(iter(ENHANCEMENTS)))
        ttk.Label(row,text='增强').pack(side='left',padx=(18,6))
        ttk.Combobox(row,textvariable=self.enhancement,values=list(ENHANCEMENTS),state='readonly',width=24).pack(side='left')
        actions=ttk.Frame(page); actions.pack(fill='x',pady=(14,12))
        self.process_button=ttk.Button(actions,text='开始处理',style='Accent.TButton',command=self.process)
        self.process_button.pack(side='left')
        ttk.Button(actions,text='手动标四角',command=self.start_manual).pack(side='left',padx=8)
        ttk.Button(actions,text='清除标点',command=self.clear_points).pack(side='left')
        self.save_button=ttk.Button(actions,text='保存全部结果',command=self.save,state='disabled')
        self.save_button.pack(side='left',padx=8)
        self.view=tk.StringVar(value='黑白扫描')
        view=ttk.Combobox(actions,textvariable=self.view,values=list(VIEWS),state='readonly',width=13)
        view.pack(side='right'); view.bind('<<ComboboxSelected>>',lambda e:self.render())
        ttk.Label(actions,text='结果视图').pack(side='right',padx=10)
        panels=ttk.Frame(page); panels.pack(fill='both',expand=True)
        panels.columnconfigure(0,weight=1,uniform='panels'); panels.columnconfigure(1,weight=1,uniform='panels'); panels.rowconfigure(1,weight=1)
        ttk.Label(panels,text='原始照片 / 四角定位',font=('Microsoft YaHei UI',11,'bold')).grid(row=0,column=0,sticky='w',pady=(0,7))
        ttk.Label(panels,text='处理结果',font=('Microsoft YaHei UI',11,'bold')).grid(row=0,column=1,sticky='w',padx=10,pady=(0,7))
        self.before=tk.Canvas(panels,bg='#e5eaf2',highlightthickness=0,cursor='crosshair')
        self.after=tk.Canvas(panels,bg='#e5eaf2',highlightthickness=0)
        self.before.grid(row=1,column=0,sticky='nsew',padx=(0,6)); self.after.grid(row=1,column=1,sticky='nsew',padx=(6,0))
        self.before.bind('<Button-1>',self.click_corner)
        self.before.bind('<Configure>',lambda e:self.render()); self.after.bind('<Configure>',lambda e:self.render())
        self.status=tk.StringVar(value='请选择照片。')
        ttk.Label(page,textvariable=self.status,wraplength=1150).pack(anchor='w',pady=(14,5))
        self.details=tk.StringVar(value='')
        ttk.Label(page,textvariable=self.details,foreground='#68778c',wraplength=1150).pack(anchor='w')
        ttk.Label(page,text='手动模式：从左上角开始沿纸边顺时针点击四角，再开始处理。自动定位结果请结合原图检查。',
                  foreground='#68778c',wraplength=1150).pack(anchor='w',pady=(10,0))
        root.bind('<Control-o>',lambda e:self.open_file()); root.protocol('WM_DELETE_WINDOW',self.close)
        if self.files:
            self.file_var.set(self.files[0].name)
            self.load(self.files[0])

    def load(self,path):
        if self.busy:return
        try:
            image=load_image(path)
        except Exception as exc:
            messagebox.showerror('无法读取照片',str(exc),parent=self.root); return
        self.path=Path(path); self.image=image; self.result=None; self.points=[]; self.manual=False
        self.save_button.configure(state='disabled'); self.details.set(f'{self.path.name}  ·  {image.shape[1]} × {image.shape[0]}')
        self.status.set('照片已载入。可自动处理，或先手动标四角。'); self.render()

    def open_file(self):
        if self.busy:return
        path=filedialog.askopenfilename(parent=self.root,title='选择文档照片',initialdir=ROOT/'dataset',
                                       filetypes=[('图片','*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),('所有文件','*.*')])
        if path:self.load(path)

    def select_dataset(self,event=None):
        if not self.busy:self.load(ROOT/'dataset'/self.file_var.get())

    def start_manual(self):
        if self.busy or self.image is None:return
        self.manual=True; self.points=[]
        self.status.set('手动标角：请依次点击左上、右上、右下、左下角（0/4）。'); self.render()

    def clear_points(self):
        if self.busy:return
        self.manual=False; self.points=[]; self.status.set('手动标点已清除，下次处理将自动定位。'); self.render()

    def click_corner(self,event):
        if self.busy or not self.manual or self._transform is None or len(self.points)>=4:return
        x0,y0,scale=self._transform
        x,y=(event.x-x0)/scale,(event.y-y0)/scale
        if 0<=x<self.image.shape[1] and 0<=y<self.image.shape[0]:
            self.points.append([float(x),float(y)])
            self.status.set(f'手动标角（{len(self.points)}/4）。'+('已齐，可开始处理。' if len(self.points)==4 else '请继续沿纸边顺时针点击。'))
            self.render()

    def process(self):
        if self.busy:return
        if self.image is None:
            messagebox.showinfo('选择照片','请先打开一张文档照片。',parent=self.root); return
        if self.manual and len(self.points)!=4:
            messagebox.showinfo('四角未齐',f'当前 {len(self.points)}/4 个角点，请继续标记或清除标点。',parent=self.root);return
        dc=DetectionConfig(mode=DETECTORS[self.detector.get()])
        ec=EnhancementConfig(mode=ENHANCEMENTS[self.enhancement.get()])
        self.busy=True; self.process_button.configure(state='disabled'); self.save_button.configure(state='disabled')
        self.status.set('正在定位、透视校正和增强…')
        future=self.pool.submit(process_document,self.image.copy(),dc,ec,list(self.points) if self.manual else None)
        self.root.after(40,lambda:self.poll(future,'process'))

    def save(self):
        if self.busy or self.result is None:return
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out=ROOT/'output'/'gui'/f'{self.path.stem}_{stamp}'
        self.busy=True; self.process_button.configure(state='disabled'); self.save_button.configure(state='disabled')
        self.status.set('正在保存各阶段结果…')
        future=self.pool.submit(export_result,self.result,out)
        self.root.after(40,lambda:self.poll(future,'save',out))

    def poll(self,future,kind,out=None):
        if not future.done():
            self.root.after(40,lambda:self.poll(future,kind,out)); return
        self.busy=False; self.process_button.configure(state='normal')
        try:
            value=future.result()
            if kind=='process':
                self.result=value
                m=value.metadata; t=m['timing_ms']
                self.status.set('处理完成。请检查左侧定位边界，并切换右侧视图查看文字细节。')
                self.details.set(f"{'手动四角' if m['corner_source']=='manual' else '自动候选'}  ·  展正 {m['rectified_size'][0]} × {m['rectified_size'][1]}  ·  总耗时 {t['total']:.1f} ms  ·  增强 {t['enhancement']:.1f} ms")
                self.render()
            else:self.status.set(f'结果已保存：{out}')
        except Exception as exc:
            self.result=None
            self.status.set('未能完成处理；可以检查照片或尝试手动标四角。')
            messagebox.showerror('处理提示',str(exc),parent=self.root)
        self.save_button.configure(state='normal' if self.result is not None else 'disabled')

    def draw(self,canvas,array,key):
        canvas.delete('all')
        if array is None:
            canvas.create_text(max(canvas.winfo_width(),1)/2,max(canvas.winfo_height(),1)/2,
                               text='处理结果将在这里显示',fill='#7a8798',font=('Microsoft YaHei UI',12));return
        rgb=cv2.cvtColor(array,cv2.COLOR_BGR2RGB) if array.ndim==3 else array
        im=Image.fromarray(rgb)
        width,height=max(20,canvas.winfo_width()-20),max(20,canvas.winfo_height()-20)
        scale=min(width/im.width,height/im.height)
        fitted=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
        x,y=(canvas.winfo_width()-fitted.width)/2,(canvas.winfo_height()-fitted.height)/2
        self._photos[key]=ImageTk.PhotoImage(fitted)
        canvas.create_image(x,y,image=self._photos[key],anchor='nw')
        if key=='before':
            self._transform=(x,y,scale)
            if self.manual:
                displayed=[(x+px*scale,y+py*scale) for px,py in self.points]
                if len(displayed)>1:
                    flat=[v for p in displayed for v in p]
                    if len(displayed)==4:flat.extend(displayed[0])
                    canvas.create_line(*flat,fill='#f56b3b',width=2)
                for i,(px,py) in enumerate(displayed):
                    canvas.create_oval(px-5,py-5,px+5,py+5,fill='#f56b3b',outline='white')
                    canvas.create_text(px+12,py-10,text=str(i+1),fill='#e64a20',font=('Segoe UI',12,'bold'))

    def render(self):
        if not hasattr(self,'before'):return
        before=self.result.detected if self.result is not None and not self.manual else self.image
        after=getattr(self.result,VIEWS[self.view.get()]) if self.result is not None else None
        self.draw(self.before,before,'before'); self.draw(self.after,after,'after')

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True); self.root.destroy()


def main():
    root=tk.Tk(); DocumentScannerApp(root); root.mainloop()

if __name__=='__main__':main()
