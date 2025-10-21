import vtk

# Step 1: Read the NRRD file using VTK
reader = vtk.vtkNrrdReader()
reader.SetFileName(r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 1\Segmentation_left.nrrd")
reader.Update()

# Step 2: Apply marching cubes to extract the surface
contour = vtk.vtkMarchingCubes()
contour.SetInputConnection(reader.GetOutputPort())
contour.SetValue(0, 1)  # Use 1 if it's a binary mask; adjust if needed
contour.Update()

# Optional: Smooth the surface
smoother = vtk.vtkSmoothPolyDataFilter()
smoother.SetInputConnection(contour.GetOutputPort())
smoother.SetNumberOfIterations(20)
smoother.SetRelaxationFactor(0.1)
smoother.FeatureEdgeSmoothingOff()
smoother.BoundarySmoothingOn()
smoother.Update()

# Step 3: Write the mesh to VTP format
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("output.vtp")
writer.SetInputConnection(smoother.GetOutputPort())
writer.Write()