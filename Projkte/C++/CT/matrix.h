#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <string>

using namespace std;

class matrix {

	vector<double> m_Mat;
	int m_nRow;
	int m_nCol;

public:

	// Constructor and Destructor
	matrix(int nrow=0, int ncol=0);
	virtual ~matrix();

	// Assginment operator: m(i,j) gives access to the element in the i-th row and j-th column
	double& operator() (int i, int j) { return m_Mat[(i-1)*m_nCol+j-1]; }

	// Load a matrix from a space-separated text (ASCII) file with filename "fName"
	bool loadFromFile(string fName);

	// Store the current contents of the matrix in a space-separated text (ASCII) file with filename "fName"
	bool storeMatrixInTextFile( string fName );

	// Print current matrix content on the screen
	void print(void);

	// Return the number of matrix columns
	int getNCols(void) { return m_nCol; };

	// Return the number of matrix rows
	int getNRows(void) { return m_nRow; };

	// Resize matrix to nrow x ncol and set elements to zero
	void reset(int nrow, int ncol);

	// Set the matrix element at location ("row","col") to "value"
	void set(int row, int col, double value);
};

#endif /* MATRIX_H_ */



