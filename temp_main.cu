{

  if (argc != 3) {
    fprintf(stderr, "usage: %s input(pbm) output(pgm)\n", argv[0]);
    return 1;
  }
  char *input_name = argv[1];
  char *output_name = argv[2];
  image<uchar> *input = loadPGM(input_name);
//------------Basic DT ------//
 image<float> *out = dt(input);
  int height = input-> height();
  int width = input->width();
for (int y = 0; y < out->height(); y++) {
    for (int x = 0; x < out->width(); x++) {
      imRef(out, x, y) = sqrt(imRef(out, x, y));
    }
  }
  image<uchar> *gray = imageFLOATtoUCHAR(out);
//-----------------------------//
  int N = width*height;
/*  
  dtype2 *h_idata, *h_odata, h_cpu;
 
  dtype2 *d_idata, *d_odata;	

*/


  dtype *h_idata, *h_odata, h_cpu;
  dtype *d_idata, *d_odata;	

  image<dtype> *input_float = imageUCHARtoFLOAT(input);
  image<dtype> *output_img = new image<dtype>(width, height, false);

  


  h_idata = (dtype*) malloc (N * sizeof (dtype));
  h_odata = (dtype*) malloc (N * sizeof (dtype));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata,N * sizeof (dtype)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, N * sizeof (dtype)));



/* //Switch to this in case of dtype2
  h_idata = (dtype2*) malloc (N * sizeof (dtype2));
  h_odata = (dtype2*) malloc (N * sizeof (dtype2));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata,N * sizeof (dtype2)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, N * sizeof (dtype2)));
