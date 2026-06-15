import tkinter as tk
from tkinter import ttk
import PyTac3D
import numpy as np
import time
import open3d as o3d

version = '3.3.0'

config_table = { 'A1':      {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 1.0 },
                 'AD2':     {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 1.0 },
                 'HDL1':    {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 1.0 },
                 'DL1':     {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 1.0 },
                 'DM1':     {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 0.8 },
                 'DS1':     {'mesh': (16,16), 'scaleF': 30, 'scaleD': 5, 'scaleN': 0.7 },
                 'DSt1':    {'mesh': (16,16), 'scaleF': 30, 'scaleD': 5, 'scaleN': 0.7 },
                 'B1':      {'mesh': (16,16), 'scaleF': 30, 'scaleD': 5, 'scaleN': 0.7 },
                 'UNKNOWN': {'mesh': (20,20), 'scaleF': 30, 'scaleD': 5, 'scaleN': 1.0 },
                 }

def getModelName(SN):
    model = SN.split('-')[0]
    if model[0] == 'Y':
        model = model[1:]

    if model in config_table.keys():
        return model
    else:
        return 'UNKNOWN'

def getDisplayConfig(SN):
    return config_table[getModelName(SN)]
    
class MainWindow:
    def __init__(self, port = 9988):
        self.root = tk.Tk()
        self.root.title('PyTac3D Displayer v3.3.0')
        self.root.resizable(0,0)
        self.root.minsize(width=400, height=200)
        self.root.grid_columnconfigure(0, weight=1)
        self.dataCache = {'STOP': None}
        self.running = True        
        self.sensorViewList = []
        self.button1 = tk.Button(self.root, text='Add View', command=self.appendSensorView)
        self.button1.pack(padx=4, pady=4,fill='x')
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.Tac3D = PyTac3D.Sensor(self.recvCallback, port=port)
        self.appendSensorView()
        
    def appendSensorView(self):
        view = SensorView(self.root, 'View %d' % (len(self.sensorViewList)+1), self.dataCache, self.Tac3D)
        SN_list = list(self.dataCache.keys())
        view.comboSN.set(SN_list[-1])
        self.sensorViewList.append(view)
                
    def recvCallback(self, frame, param):
        SN = frame.get('SN')
        if not SN in self.dataCache.keys():
            self.dataCache[SN] = frame
            for view in self.sensorViewList:
                view.updateSensorList()
        else:
            self.dataCache[SN] = frame
            
    def run(self):
        # 手动处理事件循环
        while self.running:
            time.sleep(0.015)
            self.root.update()
            time.sleep(0.015)
            for view in self.sensorViewList:
                view.update()

    def stop(self):
        self.running = False
        for view in self.sensorViewList:
            view.stop()
    
class SensorView:
    def __init__(self, root, viewName, dataCache, Tac3D):
        self.Tac3D = Tac3D
        self.SN = ''
        self.lastSN = ''
        self.config = None
        self.triangles = None
        self.dataCache = dataCache
        self.root = root
        self.viewName = viewName
        self.flags_updateSN = False

        self.enablePoint = False
        self.enableMesh = True
        self.fieldList = ['Disable', '3D_Forces', '3D_Displacements', '3D_Normals']
        self.displayField = 'Disable'
        
        self.mesh = None
        self.field = None
        self.points = None
        
        self.frame = tk.LabelFrame(root, text=viewName)
        self.frame.pack(padx=4, pady=4,fill='x')
        self.frame.grid_columnconfigure(1, weight=1)
        currRow = 0
        self.labelSN = tk.Label(self.frame, text='SN')
        self.labelSN.grid(row=currRow,column=0, sticky='w e', padx=4, pady=4)
        self.comboSN = ttk.Combobox(self.frame)
        self.comboSN.grid(row=currRow,column=1, sticky='w e', padx=4, pady=4)
        self.comboSN['state'] = 'readonly'

        currRow += 1
        self.labelMesh = tk.Label(self.frame, text='Mesh')
        self.labelMesh.grid(row=currRow,column=0, sticky='w e', padx=4, pady=4)
        self.buttonMesh = tk.Button(self.frame, text='ON', command=self.meshSwitch)
        self.buttonMesh.grid(row=currRow,column=1, sticky='w e', padx=4, pady=4)
        
        currRow += 1
        self.labelPoint = tk.Label(self.frame, text='Point Cloud')
        self.labelPoint.grid(row=currRow,column=0, sticky='w e', padx=4, pady=4)
        self.buttonPoint = tk.Button(self.frame, text='OFF', command=self.pointSwitch)
        self.buttonPoint.grid(row=currRow,column=1, sticky='w e', padx=4, pady=4)
        
        currRow += 1
        self.labelField = tk.Label(self.frame, text='Field')
        self.labelField.grid(row=currRow,column=0, sticky='w e', padx=4, pady=4)
        self.buttonField = tk.Button(self.frame, text=self.displayField, command=self.fieldSwitch)
        self.buttonField.grid(row=currRow,column=1, sticky='w e', padx=4, pady=4)
        
        currRow += 1
        self.buttonCali = tk.Button(self.frame, text='Calibrate', command=self.calibrate)
        self.buttonCali.grid(row=currRow,column=0, columnspan=2, sticky='w e', padx=4, pady=4)
        
        self.updateSensorList()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(self.viewName, 800, 600)
        self.render_option = self.vis.get_render_option()
        self.render_option.mesh_show_back_face = True
        self.render_option.point_size = 3

    def genMesh(self, meshSize):
        nx, ny = meshSize
        self.triangles = []
        for iy in range(ny-1):
            for ix in range(nx-1):
                idx = iy * nx + ix
                self.triangles.append([idx, idx+1, idx+nx])
                self.triangles.append([idx+nx+1, idx+nx, idx+1])
        self.triangles = np.array(self.triangles)
    
    def updateVisualization(self):
        self.SN = self.comboSN.get()
        frame = self.dataCache.get(self.SN)
        if frame is None:
            return
        if self.SN != self.lastSN:
            self.lastSN = self.SN
            self.config = getDisplayConfig(self.SN)
            self.genMesh(self.config['mesh'])
            
            if not self.mesh is None:
                self.vis.remove_geometry(self.mesh)
            if not self.field is None:
                self.vis.remove_geometry(self.field)
            if not self.points is None:
                self.vis.remove_geometry(self.points)
                
            self.mesh = None
            self.field = None
            self.points = None

        self.vertices = frame.get('3D_Positions')
        if self.enableMesh:
            if not self.vertices is None:
                if not self.mesh is None:
                    self.mesh.vertices = o3d.utility.Vector3dVector(self.vertices.copy())
                    self.mesh.compute_vertex_normals()
                    self.vis.update_geometry(self.mesh)
                else:
                    self.mesh = o3d.geometry.TriangleMesh()
                    self.mesh.vertices = o3d.utility.Vector3dVector(self.vertices.copy())
                    self.mesh.triangles = o3d.utility.Vector3iVector(self.triangles.copy())
                    self.mesh.compute_vertex_normals()
                    self.mesh.paint_uniform_color([0.65, 0.6, 0.75])
                    self.vis.add_geometry(self.mesh)
        elif not self.mesh is None:
            self.vis.remove_geometry(self.mesh)
            self.mesh = None
            
        if self.enablePoint:
            if not self.vertices is None:
                if not self.points is None:
                    self.points.points = o3d.utility.Vector3dVector(self.vertices.copy())
                    self.vis.update_geometry(self.points)
                else:
                    self.points = o3d.geometry.PointCloud()
                    self.points.points = o3d.utility.Vector3dVector(self.vertices.copy())
                    self.points.paint_uniform_color([0, 0, 0])
                    self.vis.add_geometry(self.points)
        elif not self.points is None:
            self.vis.remove_geometry(self.points)
            self.points = None
        
        if self.displayField != 'Disable':
            if self.displayField == '3D_Forces':
                field = frame.get('3D_Forces')
                scale = self.config['scaleF']
            elif self.displayField == '3D_Displacements':
                field = frame.get('3D_Displacements')
                scale = self.config['scaleD']
            elif self.displayField == '3D_Normals':
                field = frame.get('3D_Normals')
                scale = self.config['scaleN']

            if not self.vertices is None and not field is None:
                ends = self.vertices + field * scale
                if not self.field is None:
                    self.field.points = o3d.utility.Vector3dVector(np.vstack([self.vertices, ends]))
                    self.field.lines = o3d.utility.Vector2iVector([[i, i + len(self.vertices)] for i in range(len(self.vertices))])
                    self.vis.update_geometry(self.field)
                else:
                    self.field = o3d.geometry.LineSet()
                    self.field.points = o3d.utility.Vector3dVector(np.vstack([self.vertices, ends]))
                    self.field.lines = o3d.utility.Vector2iVector([[i, i + len(self.vertices)] for i in range(len(self.vertices))])
                    self.field.paint_uniform_color([0, 0, 0.8])
                    self.vis.add_geometry(self.field)
            elif not self.field is None:
                self.vis.remove_geometry(self.field)
                self.field = None
        elif not self.field is None:
            self.vis.remove_geometry(self.field)
            self.field = None
        
    def meshSwitch(self):
        if self.enableMesh:
            self.enableMesh = False
            self.buttonMesh['text'] = 'OFF'
        else:
            self.enableMesh = True
            self.buttonMesh['text'] = 'ON'

    def pointSwitch(self):
        if self.enablePoint:
            self.enablePoint = False
            self.buttonPoint['text'] = 'OFF'
        else:
            self.enablePoint = True
            self.buttonPoint['text'] = 'ON'
            
    def fieldSwitch(self):
        idx = self.fieldList.index(self.displayField)
        idx = (idx+1) % len(self.fieldList)
        self.displayField = self.fieldList[idx]
        self.buttonField['text'] = self.displayField
        print(self.displayField)
    
    def calibrate(self):
        self.Tac3D.calibrate(self.SN)
        
    def updateSensorList(self):
        self.flags_updateSN = True
        
    def update(self):
        self.vis.poll_events()
        self.updateVisualization()
        #self.vis.update_renderer()
        if self.flags_updateSN:
            self.flags_updateSN = False
            SN_list = list(self.dataCache.keys())
            self.comboSN['values'] = SN_list
            if len(SN_list) == 2:
                self.comboSN.set(SN_list[-1])

    def stop(self):
        self.vis.destroy_window()
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    else:
        port = 9988

    window = MainWindow(port)
    window.run()
    window.stop()
