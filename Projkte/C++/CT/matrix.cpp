#include "matrix.h"
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <iterator>

using namespace std;

matrix::matrix(int nrow, int ncol)
{
	m_nRow = nrow;
	m_nCol = ncol;
	m_Mat.resize(nrow*ncol);
}

matrix::~matrix() {
}


bool matrix::loadFromFile( string fName )
{

	std::string Line;
	int curRow = 0;
	int curCol = 0;

	m_nRow=0;
	m_nCol=0;

	// open file
	const char* const fNameC = fName.c_str();
	std::ifstream in( fNameC );
	if( !in )
	{
		std::cout << "can't open file: " << fName << std::endl;
		return false;
	}

	// check if number of rows and columns are already known
	if( m_nRow == 0 || m_nCol == 0 )
	{
		// read all lines in current document and count
		while( getline( in, Line ) )
		{
			if( m_nRow == 0 )
			{
				istringstream iss ( Line, istringstream::in );
				istream_iterator<double> iter( iss );
				for( ; iter!= istream_iterator<double>(); iter++ )
				{
					m_nCol++;
				}
			}
			m_nRow++;
		}
		// re-size matrix
		m_Mat.resize(m_nRow*m_nCol);

		in.close();
		in.open( fName.c_str() );
	}

	// read all lines in current document and fill matrix
	while( getline( in, Line ) )
	{
		istringstream iss ( Line, istringstream::in );
		istream_iterator<double> iter( iss );
		for( ; iter != istream_iterator<double>(); iter++ )
		{
			m_Mat[curRow*m_nCol+curCol++] = *iter;
		}
		curRow++;
		curCol = 0;
	}
	// close file again
	in.close();
	return true;
}



bool matrix::storeMatrixInTextFile( string fName )
{
	ofstream fout;
	fout.open( fName.c_str() );
	if( !fout )
	{
		std::cout << "can't open output file" << std::endl;
		return false;
	}

	for( int row = 0; row < m_nRow; row++ )
	{
		for( int col = 0; col < m_nCol; col++ )
		{
			fout << m_Mat[row*m_nCol+col] << " ";
		}
		fout << std::endl;
	}

	fout.close();
	return true;
}


void matrix::print()
{
	int i=1;

	cout << "Printing "<< m_nRow <<"x"<<m_nCol<<" matrix" <<endl;
	for( int row = 0; row < m_nRow; row++ )
	{
		for( int col = 0; col < m_nCol; col++ )
		{
			cout << m_Mat[row*m_nCol+col] << " ";
		}
		cout << std::endl;
	}
}


void matrix::reset(int nrow, int ncol) {
	m_nRow = nrow;
	m_nCol = ncol;
	m_Mat.resize(nrow*ncol);
}


void matrix::set(int row, int col, double value) 
{
	m_Mat[(row-1)*m_nCol+col-1] = value;
}
