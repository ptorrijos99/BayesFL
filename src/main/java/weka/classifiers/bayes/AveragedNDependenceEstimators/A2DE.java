/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    A2DE.java
 *    Copyright (C) 2012
 *    Algorithm developed by: Geoff Webb
 *    Code written by: Nayyar Zaidi and Janice Boughton
 */
package weka.classifiers.bayes.AveragedNDependenceEstimators;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities;
import weka.core.OptionHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
<!-- globalinfo-start -->
 * A2DE achieves highly accurate classification by averaging over all of a small space of alternative naive-Bayes-like models that have weaker (and hence less detrimental)
 * independence assumptions than naive Bayes. The resulting algorithm is computationally efficient while delivering highly accurate classification on many learning  tasks. <br/>
 * <br/>
 * For more information, see<br/>
 * <br/> G.I. Webb, J. Boughton, F. Zheng, K.M. Ting and H. Salem (2012). Learning by extrapolation from marginal to full-multivariate probability distributions: decreasingly naive {Bayesian} classification. Machine Learning. 86(2):233-272.<br/>
 * <br/>
 * Further papers are available at<br/> http://www.csse.monash.edu.au/~webb/.<br/>
 * <br/>
 * Default frequency limit set to 1.
 * <p/>
<!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Webb2012,
 * author = {Webb, Geoffrey I. and Boughton, Janice and Zheng, Fei and Ting, Kai Ming and Salem, Houssam},
 * title = {Learning by extrapolation from marginal to full-multivariate probability distributions: decreasingly naive {Bayesian} classification},
 * journal = {Machine Learning},
 * year = {2012},
 * volume = {86},
 * pages = {233-272},
 * number = {2},
 * note = {10.1007/s10994-011-5263-6},
 * affiliation = {Faculty of Information Technology, Monash University, Clayton, VIC 3800, Australia},
 * issn = {0885-6125},
 * publisher = {Springer Netherlands},
 * url = {http://dx.doi.org/10.1007/s10994-011-5263-6}
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 *
<!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -F &lt;int&gt;
 *  Impose a frequency limit for superParents
 *  (default is 1)</pre>
 *
 * <pre> -M &lt;int&gt;
 *  Specify a weight to use with m-estimate
 *  (default is 1)</pre>
 *  
 *  <pre>
 *  Use A2DEUpdateable classifier for incremental learning so that probabilities are 
 *  calculated at classification time that will speed up learning but slows 
 *  down classification. 
 *  Numeric attributes are not supported in A1DEUpdateable version.
 *  </pre>
 *  
 *  <pre> -S (Optional) &lt;int&gt;
 *  Specify critical value of specialization-generalization for 
 *  Subsumption Resolution (default is false).
 *  Results in lowering bias and increasing variance of classification. 
 *  Recommended for large training data.
 *  See: 
 *  &#64;inproceedings{Zheng2006,
 *    author = {Fei Zheng and Geoffrey I. Webb},
 *    booktitle = {Proceedings of the Twenty-third International Conference on Machine  Learning (ICML 2006)},
 *    pages = {1113-1120},
 *    publisher = {ACM Press},
 *    title = {Efficient Lazy Elimination for Averaged-One Dependence Estimators},
 *    year = {2006},
 *    ISBN = {1-59593-383-2}
 * }
 *  </pre>
 *  
 *  <pre> -W (Optional) &lt;int&gt;
 *  Weighted A2DE. Uses mutual information between parent attributes and the class as weight of
 *  each SPODE (default is false).
 *  Results in lowering bias and increasing variance of classification.
 *  Recommended for large training data.
 *  Can not use weighting for A1DEUpdateable classifier.
 *  See:  
 *  &#64;inproceedings{Jiang2006,
 *    author = {L. Jiang and H. Zhang},
 *    booktitle = {Proceedings of the 9th Biennial Pacific Rim International Conference on Artificial Intelligence, PRICAI 2006},
 *    pages = {970-974},
 *    series = {LNAI},
 *    title = {Weightily Averaged One-Dependence Estimators},
 *    volume = {4099},
 *    year = {2006}
 * }
 *  </pre>
 *  
 *
 * <!-- options-end -->
 *
 * @author Nayyar Zaidi (nayyar.zaidi@monash.edu)
 * @author Janice Boughton (jrbought@csse.monash.edu.au)
 * @version $Revision: 2 $
 */

public class A2DE extends AbstractClassifier 
implements OptionHandler, WeightedInstancesHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = 9197439980415113523L;

	/** The discretization filter  */
	protected weka.filters.supervised.attribute.Discretize m_Disc = null;

	/**
	 * The frequency of three attribute-values occurring together (i.e., two
	 * parents and one child) for each class. Only unique combinations are stored.
	 */
	private double[] m_3vCondiCounts;

	/**
	 * The frequency of two attribute-values occurring together for each class.
	 * Only unique combinations are stored.
	 */
	private double[] m_2vCondiCounts;

	/** The frequency of each attribute value for each class */
	private double[] m_1vCondiCounts;

	/**
	 * The frequency of three attribute-values occurring together (i.e., two
	 * parents and one child) for each class. Only unique combinations are stored.
	 */
	private double[] m_3vCondiCountsP1;

	/**
	 * The frequency of three attribute-values occurring together (i.e., two
	 * parents and one child) for each class. Only unique combinations are stored.
	 */
	private double[] m_3vCondiCountsP2;

	/**
	 * The frequency of two attribute-values occurring together for each class.
	 * Only unique combinations are stored.
	 */
	private double[] m_2vCondiCountsOpp;

	/**
	 * The frequency of two attribute-values occurring together for each class.
	 * Only unique combinations are stored.
	 */
	private double[] m_2vCondiCountsJoint;

	/** The frequency of each attribute value for each class */
	private double[] m_1vCondiCountsNB;

	/**
	 * The frequency of two attribute values (as parents and no child) for the
	 * dataset. Note: m_2vCondicounts also has the frequency of two attribute
	 * values but per class.
	 */
	private double[][] m_Frequencies2p;

	/**
	 * The frequency of each attribute value (as parent and no child) for the
	 * dataset (all classes). Note: m_1vCondiCounts also has the frequency of an
	 * attribute but per class.
	 */
	private double[] m_Frequencies;

	/**
	 * Offsets to index combinations with 3 attribute values. An attribute-value
	 * is chosen for this level of offsets if it's the largest from a combination
	 * of three.
	 * E.g.:  m_3vOffsets[9] + m_2vOffsets[6] + 1;
	 */
	private int[] m_3vOffsets;

	/**
	 * Offsets to index combinations with 2 or more attribute values. An
	 * attribute-value is chosen for this level of offsets
	 * if it's the largest from a combination
	 * of two (or the two remaining from a larger combination)
	 * E.g.: 
	 * Three values: m_3vOffsets[9] + m_2vOffsets[6] + 1;
	 * Two values: m_2vOffsets[7] + 1;
	 */
	private int[] m_2vOffsets;

	/** The number of times each class value occurs in the dataset */
	private double[] m_ClassCounts;

	/** The m-Estimate of the probabilities of each class */
	private double[] m_ClassProbabilities;

	/** The sums of attribute-class counts  
	 *    -- if there are no missing values for att, then m_SumForCounts[classVal][att] 
	 *       will be the same as m_ClassCounts[classVal] 
	 */
	private double[][] m_SumForCounts;

	/** The number of classes */
	private int m_NumClasses;

	/** The number of attributes in dataset, including class */
	private int m_NumAttributes;

	/** The number of instances in the dataset */
	private int m_NumInstances;

	/** The index of the class attribute */
	private int m_ClassIndex;

	/** The dataset */
	private Instances m_Instances;

	/**
	 * The total number of values (including an extra for each attribute's 
	 * missing value, which are included in m_CondiCounts) for all attributes 
	 * (not including class).  Eg. for three atts each with two possible values,
	 * m_TotalAttValues would be 9 (6 values + 3 missing).
	 * This variable is used when allocating space for m_CondiCounts matrix.
	 */
	private int m_TotalAttValues;

	/** The starting index (in m_CondiCounts matrix) of values for each att */
	private int[] m_StartAttIndex;

	/** The number of values for each attribute */
	private int[] m_NumAttValues;

	/** The number of valid class values observed in dataset 
	 *  -- with no missing classes, this number is the same as m_NumInstances.
	 */
	private int m_SumInstances;

	/** (Input paramters) An att's frequency must be this value or more to be a superParent */
	private int m_Limit = 1;

	/** (Input paramters) value for m in m-estimate */
	private double m_Weight = 1;
	
	/** Initialize SPODE probabilities to some value to avoid underflow or overflow */
	private double probInitializer = 1;

	/** Initialize SPODE probabilities to some value to avoid underflow or overflow */
	private double probInitializerAODE = 1;

	/** Initialize SPODE probabilities to some value to avoid underflow or overflow */
	private double probInitializerA2DE = 1;

	/** (Input paramters) Do Subsumption Resolution */
	private boolean m_SubsumptionResolution = false;

	/** the critical value for the specialization-generalization */
	private int m_Critical = 100;

	/** (Input paramters) Do Weighted A2DE */
	private boolean m_WeightedA2DE = false;

	/** The array of mutual information between each attribute and class */
	private double[][] m_mutualInformation;
	
	/** Calculate conditional probability at training time or testing time. */
	protected static boolean m_Incremental = false;

	/** Use Discretization. */
	protected static boolean m_UseDiscretization = true;


	/**
	 * Returns a string describing this classifier
	 * 
	 * @return a description of the classifier suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "A2DE achieves highly accurate classification by averaging over "
		+ "all of a small space of alternative naive-Bayes-like models that have "
		+ "weaker (and hence less detrimental) independence assumptions than "
		+ "naive Bayes. The resulting algorithm is computationally efficient "
		+ "while delivering highly accurate classification on many learning  "
		+ "tasks.\n\n"
		+ "For more information, see\n\n" + getTechnicalInformation().toString() 
		+ "\n\n"
		+ "Further papers are available at\n"
		+"  http://www.csse.monash.edu.au/~webb/.\n\n"
		+ "Use m-estimate for smoothing base probability estimates with" 
		+ "a default of 1 (m value can changed via option -M).\n  "
		+ "Default mode is non-incremental that is probabilites are computed "
		+ "at learning time. An incremental version can be used via option -I. \n"
		+ "Default frequency limit set to 1.";
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Webb, Geoffrey I. and Boughton, Janice and Zheng, Fei and Ting, Kai Ming and Salem, Houssam");
		result.setValue(Field.YEAR, "2012");
		result.setValue(Field.TITLE,
		"Learning by extrapolation from marginal to full-multivariate probability distributions: decreasingly naive {Bayesian} classification");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "86");
		result.setValue(Field.NUMBER, "2");
		result.setValue(Field.PAGES, "233-272");			

		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data
	 * @exception Exception if the classifier has not been generated
	 * successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		m_Instances = new Instances(instances);
		m_Instances.deleteWithMissingClass();

		// Discretize instances if required
		if (m_UseDiscretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setInputFormat(m_Instances);
			m_Instances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
		}

		// reset variable for this fold
		m_SumInstances = 0;
		m_NumInstances = m_Instances.numInstances();
		m_NumClasses = instances.numClasses();
		m_ClassIndex = instances.classIndex();
		m_NumAttributes = instances.numAttributes();

		// allocate space for attribute reference arrays
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];
		
		m_TotalAttValues = 0;
		for(int i = 0; i < m_NumAttributes; i++) {
			if(i != m_ClassIndex) {
				m_StartAttIndex[i] = m_TotalAttValues;
				m_NumAttValues[i] = m_Instances.attribute(i).numValues();
				m_TotalAttValues += m_NumAttValues[i] + 1;
				// + 1 so room for missing value count
			} else {
				// m_StartAttIndex[i] = -1;  // class isn't included
				m_NumAttValues[i] = m_NumClasses;
			}
		}

		m_3vOffsets = new int[m_TotalAttValues];
		m_2vOffsets = new int[m_TotalAttValues];


		//Calculate and store offsets for attribute values.

		//Track the index of two-valued combinations and accumulate their size
		int nextInnerIndex = 0, innerAdd = 0, totalInner = 0;

		//Track the index of four-valued combinations and accumulate their size
		int nextOuterIndex = 0, outerAdd = 0;

		int curAtt = 0;
		for (int i = 0; i < m_NumAttributes; i++) {
			if (i != m_ClassIndex){
				for (int j = 0; j < m_NumAttValues[i] + 1; j++) { // each attribute-value
					// and missing value
					m_3vOffsets[curAtt] = nextOuterIndex;
					m_2vOffsets[curAtt] = nextInnerIndex;

					// work out the offsets for the *next* attribute value
					nextOuterIndex = m_3vOffsets[curAtt] + outerAdd;
					nextInnerIndex = m_2vOffsets[curAtt] + innerAdd;
					curAtt++;
					totalInner += innerAdd;
				}
				innerAdd += m_NumAttValues[i] + 1; // +1 for missing value
				outerAdd = totalInner;
			}
		}

		//If an offset starts at n, then there are n-1 locations before it
		//Since we work out the offset of the attribute value following the 
		//current one, after the last loop we have:
		//nextOuter = number of three-valued combinations
		//nextInner = number of two-valued combinations

		/* Allocate space for counts C(.,.) and if non-incremental Classifier 
		 * is checked compute probabilities P(.,.)		 
		 */

		/*
		 *    m_3vCondiCounts in Incremental Version:
		 *     ------------- ------------- ------------- ------------- 
		 *    | C(x3,x2,x1) | C(x4,x3,x2) | C(x4,x3,x1) | C(x4,x2,x1) |
		 *     ------------- ------------- ------------- -------------
		 *  
		 *    m_2vCondiCounts in Incremental Version: 
		 *     ---------- ---------- ---------- ---------- ---------- ----------
		 *    | C(x2,x1) | C(x3,x2) | C(x3,x1) | C(x4,x3) | C(x4,x2) | C(x4,x1) |
		 *     ---------- ---------- ---------- ---------- ---------- ----------
		 *
		 *    m_1vCondiCounts in Incremental Version: 
		 *     ------- ------- ------- -------    
		 *    | C(x1) | C(x2) | C(x3) | C(x4) |
		 *     ------- ------- ------- -------  
		 */
		m_3vCondiCounts = new double[nextOuterIndex * m_NumClasses];
		m_2vCondiCounts = new double[nextInnerIndex * m_NumClasses];
		m_1vCondiCounts = new double[m_TotalAttValues * m_NumClasses];

		// Additional data structure used in the program
		m_ClassCounts = new double[m_NumClasses];
		m_SumForCounts = new double[m_NumClasses][m_NumAttributes];
		m_Frequencies = new double[m_TotalAttValues];
		m_Frequencies2p = new double[m_TotalAttValues][m_TotalAttValues];


		// calculate the counts
		for(int k = 0; k < m_NumInstances; k++) {
			addToCounts(m_Instances.instance(k));
		}

		// allocate memory for mutual information between attribute and class
		m_mutualInformation = new double[m_NumAttributes][m_NumAttributes];
		for (int i = 0; i < m_NumAttributes; i++) {
			for (int j = 0; j < m_NumAttributes; j++) {
				m_mutualInformation[i][j] = 1;
			}
		}
			
		if (m_WeightedA2DE && !m_Incremental) {
			/*
			 * Weighted A2DE Flag is set. 
			 * Compute mutual information between each attribute and class 
			 */
			boolean nonZeroFlag = true;
			for (int att1 = m_NumAttributes - 1; att1 >= 0; att1--) {
				if (att1 == m_ClassIndex) continue;

				for (int att2 = att1 - 1; att2 >= 0; att2--) {
					if (att2 == m_ClassIndex) continue;

					m_mutualInformation[att1][att2] = mutualInfo(att1, att2);
					//System.out.println("mutualInfo(" + att1 + "," + att2 + ") = " + m_mutualInformation[att1][att2]);
					
					if (m_mutualInformation[att1][att2] != 0) {
						nonZeroFlag = false;
					}
				}
			}

			if (nonZeroFlag) {
				for (int att1 = 0; att1 < m_NumAttributes; att1++) {
					if (att1 == m_ClassIndex) continue;
					for (int att2 = 0; att2 < m_NumAttributes; att2++) {
						if (att2 == m_ClassIndex) continue;
						m_mutualInformation[att1][att2] = (double)1/m_NumClasses;
					}
				}
			}

			//Utils.normalize(m_mutualInformation);
		}

		probInitializer = Double.MAX_VALUE;
		probInitializerAODE = Double.MAX_VALUE/m_NumAttributes;
		probInitializerA2DE = Double.MAX_VALUE/(m_NumAttributes * m_NumAttributes);

		// Calculate conditional probability at training time
		if (!m_Incremental) {
			/*
			 *    Following probabilities are also conditioned on y
			 *    
			 *     m_3vCondiCounts in Incremental Version:
			 *     ------------- ------------- ------------- ------------- 
			 *    | C(x3|x2,x1) | C(x4|x3,x2) | C(x4|x3,x1) | C(x4|x2,x1) |
			 *     ------------- ------------- ------------- -------------
			 *     
			 *     m_3vCondiCountsOppP1 in Incremental Version:
			 *     ------------- ------------- ------------- ------------- 
			 *    | C(x2|x3,x1) | C(x3|x4,x2) | C(x3|x4,x1) | C(x2|x4,x1) |
			 *     ------------- ------------- ------------- -------------
			 *     
			 *     m_3vCondiCountsOppP2 in Incremental Version:
			 *     ------------- ------------- ------------- ------------- 
			 *    | C(x1|x3,x2) | C(x2|x4,x3) | C(x1|x4,x3) | C(x1|x4,x2) |
			 *     ------------- ------------- ------------- -------------
			 *    
			 *    m_2vCondiCounts in Incremental Version: 
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *    | P(x2|x1) | P(x3|x2) | P(x3|x1) | P(x4|x3) | P(x4|x2) | P(x4|x1) |
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *     
			 *    m_2vCondiCountsOpp in Incremental Version: 
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *    | P(x1|x2) | P(x2|x3) | P(x1|x3) | P(x3|x4) | P(x2|x4) | P(x1|x4) |
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *     
			 *    m_2vCondiCountsJoint in Incremental Version: 
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *    | P(x1,x2) | P(x2,x3) | P(x1,x3) | P(x3,x4) | P(x2,x4) | P(x1,x4) |
			 *     ---------- ---------- ---------- ---------- ---------- ----------
			 *
			 *    m_1vCondiCounts in Incremental Version: 
			 *     --------- --------- --------- --------- ---------
			 *    | P(x1,y) | P(x2,y) | P(x3,y) | P(x4,y) | P(x5,y) |
			 *     --------- --------- --------- --------- ---------
			 *     
			 *    m_1vCondiCountsNB in Incremental Version: 
			 *     --------- --------- --------- --------- ---------
			 *    | P(x1|y) | P(x2|y) | P(x3|y) | P(x4|y) | P(x5|y) |
			 *     --------- --------- --------- --------- ---------
			 */			
			m_3vCondiCountsP1 = new double[nextOuterIndex * m_NumClasses];
			m_3vCondiCountsP2 = new double[nextOuterIndex * m_NumClasses];	

			m_2vCondiCountsOpp = new double[nextInnerIndex * m_NumClasses];
			m_2vCondiCountsJoint = new double[nextInnerIndex * m_NumClasses];

			m_1vCondiCountsNB = new double[m_TotalAttValues * m_NumClasses];

			m_ClassProbabilities = new double[m_NumClasses];

			calcConditionalProbs();
		}

		// Free up some space
		m_Instances.delete();
	}

	/**
	 * Computes mutual information between each attribute and class attribute.
	 *
	 * @param att1 is the attribute
	 * @return the conditional mutual information between son and parent given class
	 * @throws Exception 
	 */
	private double mutualInfo(int att1, int att2) throws Exception {

		double mutualInfo = 0;

		double[] PriorsClass = new double[m_NumClasses];
		double[][] PriorsAttribute = new double[m_NumAttValues[att1]][m_NumAttValues[att2]];

		double[][][] PriorsClassAttribute = new double[m_NumClasses][m_NumAttValues[att1]][m_NumAttValues[att2]];

		double missingForAtt1 = m_Frequencies[m_StartAttIndex[att1] + m_NumAttValues[att1]];
		double missingForAtt2 = m_Frequencies[m_StartAttIndex[att2] + m_NumAttValues[att2]];
		double missingForAtt1orAtt2 = missingForAtt1 + missingForAtt2 - m_Frequencies2p[m_StartAttIndex[att1] + m_NumAttValues[att1]][m_StartAttIndex[att2] + m_NumAttValues[att2]];

		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			PriorsClass[classVal] = MEsti(m_ClassCounts[classVal], m_SumInstances, m_NumClasses);			
		}

		/* Compute P(x1,x2)
		 * 
		 *                    C(x1,x2) + m/(|x1|*|x2|)
		 * 		P(x1,x2) = ---------------------------------
		 *                      (#N - ?(x1,x2))  + m 
		 * 
		 *    Store result in 
		 */
		for (int i = 0; i < m_NumAttValues[att1]; i++) {
			for (int j = 0; j < m_NumAttValues[att2]; j++) {
				PriorsAttribute[i][j] = MEsti(m_Frequencies2p[m_StartAttIndex[att1] + i][m_StartAttIndex[att2] + j], (m_SumInstances - missingForAtt1orAtt2), (m_NumAttValues[att1] * m_NumAttValues[att2]));
				//System.out.println("P(x1,x2) = (" + m_Frequencies2p[m_StartAttIndex[att1] + i][m_StartAttIndex[att2] + j] + " + m/" +  (m_NumAttValues[att1] * m_NumAttValues[att2]) + ") / (" + (m_SumInstances - missingForAtt1orAtt2) + " + m) = " + PriorsAttribute[i][j]);
			}
		}

		/* 
		 * Get the count C(x1,x2,y) from m_CondiCount and compute the joint
		 * probability as
		 * 
		 *                C(x1,x2,y) + m/(y*Card(x1)*Card(y))
		 * P(x1,x2,y) = --------------------------------------
		 *                   (N - ?(x1,x2,y)) + m
		 *   
		 */	
		double p1p2Count = 0;
		int p1val = 0, p2val = 0;
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			for (int i = 0; i < m_NumAttValues[att1]; i++) {
				for (int j = 0; j < m_NumAttValues[att2]; j++) {

					p1val = m_StartAttIndex[att1] + i;
					p2val = m_StartAttIndex[att2] + j;
					p1p2Count = getCountFromTable(classVal,p1val,p2val,p2val);
					PriorsClassAttribute[classVal][i][j] = MEsti(p1p2Count, (m_SumInstances - missingForAtt1orAtt2), (m_NumClasses * m_NumAttValues[att1] * m_NumAttValues[att2]));	
					
					//System.out.println("P(x1,x2,y) = (" + p1p2Count + " + m/" + (m_NumClasses * m_NumAttValues[att1] * m_NumAttValues[att2]) + ") / (" + (m_SumInstances - missingForAtt1orAtt2) + " + m) = " + PriorsClassAttribute[classVal][i][j]);
				}
			}
		}

		for (int c = 0; c < m_NumClasses; c++) {
			for (int i = 0; i < m_NumAttValues[att1]; i++) {
				for (int j = 0; j < m_NumAttValues[att2]; j++) {
					mutualInfo += PriorsClassAttribute[c][i][j] * log2(PriorsClassAttribute[c][i][j], PriorsClass[c] * PriorsAttribute[i][j]);
					//System.out.println(mutualInfo + " = " + PriorsClassAttribute[c][i][j] + " log2 (" + PriorsClassAttribute[c][i][j] + "/(" + PriorsClass[c] + " x " + PriorsAttribute[i][j] + ")");
				}
			}
		}

		return mutualInfo;
	}

	/**
	 * compute the logarithm whose base is 2.
	 *
	 * @param x numerator of the fraction.
	 * @param y denominator of the fraction.
	 * @return the natual logarithm of this fraction.
	 */
	private double log2(double x, double y){

		if (x < Utils.SMALL || y < Utils.SMALL)
			return 0.0;
		else
			return Math.log(x/y)/Math.log(2);
	}

	/** 
	 * This function converts the counts in m_CondiCounts to conditional 
	 * probability estimates. This method is called during model building, so
	 * the conditional probabilities don't have to be calculated at the test
	 * time.
	 * @throws Exception 
	 */
	public void calcConditionalProbs() throws Exception {

		/* Local variables */
		double p1Count = 0, p2Count = 0;
		double p1p2cCount = 0, p1p2Count = 0;
		double childCondProb = 0, p1CondProb = 0, p2CondProb = 0;
		double jointProb = 0, condProb = 0, condProbOpp;
		double missingForParent1 = 0, missingForParent2 = 0, missingForChild = 0;
		double p1cCount = 0, p2cCount = 0;

		/* 
		 * Calculate conditional probabilities from the counts such that
		 * 
		 */
		for (int p1 = m_NumAttributes - 1; p1 >= 0; p1--) { // parent 1
			if (p1 == m_ClassIndex) continue;
			for (int p1val = m_StartAttIndex[p1]; p1val < m_StartAttIndex[p1] + m_NumAttValues[p1]; p1val++) { 

				for (int p2 = p1 - 1; p2 >= 0; p2--) {  // parent 2   
					if (p2 == m_ClassIndex) continue;
					for (int p2val = m_StartAttIndex[p2]; 
					p2val < m_StartAttIndex[p2] + m_NumAttValues[p2]; p2val++) {

						for (int child = p2 - 1; child >= 0; child--) {  // child
							if (child == m_ClassIndex) continue;
							for (int childval = m_StartAttIndex[child]; 
							childval < m_StartAttIndex[child] + m_NumAttValues[child]; childval++) {

								for (int classval = 0; classval < m_NumClasses; classval++) {

									p1p2cCount = getCountFromTable(classval,p1val,p2val,childval);

									p1p2Count = getCountFromTable(classval, p1val, p2val, p2val);
									p1cCount = getCountFromTable(classval, p1val, childval, childval);
									p2cCount = getCountFromTable(classval, p2val, childval, childval);

									/* 
									 *                     C(xc,xp1,xp2,y) + m/Card(xc)
									 * P(xc|xp1,xp2,y) = --------------------------------
									 *                     (C(xp1,xp2,y) - ?(xc)) + m
									 * 
									 * where Card(xc) is cardinality of xc and 
									 *       ?(xc) are the number of missing values of xc
									 *       
									 * Store P(xc|xp1,xp2,y) in m_3vCondiCounts
									 */
									missingForChild = m_3vCondiCounts[(m_3vOffsets[p1val] + m_2vOffsets[p2val] + m_StartAttIndex[child] + m_NumAttValues[child]) * m_NumClasses + classval];
									childCondProb = (p1p2cCount + m_Weight/m_NumAttValues[child]) /	((p1p2Count - missingForChild) + m_Weight);									
									m_3vCondiCounts[(m_3vOffsets[p1val] + m_2vOffsets[p2val] + childval) * m_NumClasses + classval] = childCondProb;

									/* 
									 *                     C(xc,xp1,xp2,y) + m/Card(xp1)
									 * P(xp1|xc,xp2,y) = --------------------------------
									 *                     (C(xc,xp2,y) - ?(xp1)) + m
									 * 
									 * where Card(xp1) is cardinality of xp1 and 
									 *       ?(xp1) are the number of missing values of xp1
									 *       
									 * Store P(xp1|xc,xp2,y) in m_CondiCounts
									 */
									missingForParent1 = m_3vCondiCounts[(m_3vOffsets[m_StartAttIndex[p1] + m_NumAttValues[p1]] + m_2vOffsets[p2val] + childval) * m_NumClasses + classval];
									p1CondProb = (p1p2cCount + m_Weight/m_NumAttValues[p1]) / ((p2cCount - missingForParent1) + m_Weight);
									m_3vCondiCountsP1[(m_3vOffsets[p1val] + m_2vOffsets[p2val] + childval) * m_NumClasses + classval] = p1CondProb;

									/*
									 *                      C(xc,xp1,xp2,y) + m/Card(xp2)
									 * P(xp2|xc,xp1,y) = --------------------------------
									 *                     (C(xc,xp1,y) - ?(xp2)) + m
									 * 
									 * where Card(xp2) is cardinality of xc and 
									 *       ?(xp2) are the number of missing values of xc
									 *        
									 *       
									 * Store P(xp2|xc,xp1,y) in m_CondiCounts
									 */
									missingForParent2 = m_3vCondiCounts[(m_3vOffsets[p1val] + m_2vOffsets[m_StartAttIndex[p2] + m_NumAttValues[p2]] + childval) * m_NumClasses + classval];
									p2CondProb = (p1p2cCount + m_Weight/m_NumAttValues[p2]) / ((p1cCount - missingForParent2) + m_Weight);
									m_3vCondiCountsP2[(m_3vOffsets[p1val] + m_2vOffsets[p2val] + childval) * m_NumClasses + classval] = p2CondProb;
								} // end classval

							} // end childval
						} // end child

					} // end p2val
				}  // end p2

			}  // end p1val
		}  // end p1

		/*
		 * 
		 */	
		for (int p1 = m_NumAttributes - 1; p1 >= 0; p1--) { // parent 1
			if (p1 == m_ClassIndex) continue;

			double missingForP1 = 0;
			missingForP1 = m_Frequencies[m_StartAttIndex[p1] + m_NumAttValues[p1]];


			for (int p1val = m_StartAttIndex[p1]; p1val < m_StartAttIndex[p1] + m_NumAttValues[p1]; p1val++) {				

				for (int p2 = p1 - 1; p2 >= 0; p2--) {  // parent 2   
					if (p2 == m_ClassIndex) continue;

					double missingForP2 = 0;
					missingForP2 = m_Frequencies[m_StartAttIndex[p2] + m_NumAttValues[p2]];

					for (int p2val = m_StartAttIndex[p2]; p2val < m_StartAttIndex[p2] + m_NumAttValues[p2]; p2val++) {

						double missingForP1orP2 = missingForP1 + missingForP2 - m_Frequencies2p[m_StartAttIndex[p1] + m_NumAttValues[p1]][m_StartAttIndex[p2] + m_NumAttValues[p2]];				

						for (int classval = 0; classval < m_NumClasses; classval++) {

							p1Count = getCountFromTable(classval,p1val,p1val,p1val); 
							p2Count = getCountFromTable(classval,p2val,p2val,p2val);

							p1p2Count = getCountFromTable(classval,p1val,p2val,p2val);

							/* 
							 * 1. Compute P(xp1|xp2)
							 *  
							 *                     C(xp1,xp2,xp2) + m/|xp1|)
							 * 		P(xp1|xp2) = ----------------------------------
							 *                       (C(xp2) - ?(xp1)) + m
							 * 
							 *    Store result in 
							 */		
							double missingForP1PerClass = m_2vCondiCounts[(m_2vOffsets[m_StartAttIndex[p1] + m_NumAttValues[p1]] + p2val) * m_NumClasses + classval];
							condProb = (p1p2Count + m_Weight/m_NumAttValues[p1]) / ((p2Count - missingForP1PerClass) + m_Weight);
							m_2vCondiCounts[(m_2vOffsets[p1val] + p2val) * m_NumClasses + classval] = condProb;

							/* 
							 * 1. Compute P(xp1|xp2)
							 *  
							 *                     C(xp1,xp2,xp2) + m/|xp2|)
							 * 		P(xp2|xp1) = ----------------------------------
							 *                       (C(xp1) - ?(xp2)) + m
							 * 
							 *    Store result in 
							 */		
							double missingForP2PerClass = m_2vCondiCounts[(m_2vOffsets[p1val] + (m_StartAttIndex[p2] + m_NumAttValues[p2])) * m_NumClasses + classval]; 
							condProbOpp = (p1p2Count + m_Weight/m_NumAttValues[p2]) / ((p1Count - missingForP2PerClass) + m_Weight);
							m_2vCondiCountsOpp[(m_2vOffsets[p1val] + p2val) * m_NumClasses + classval] = condProbOpp;

							/* 2. Compute P(xp1,xp2)
							 * 
							 *                    C(xp1,xp2,y) + m/(|xp1|*|xp2|)
							 * 		P(xp1,xp2) = ---------------------------------
							 *                      (#N - ?(xp1,xp2))  + m 
							 * 
							 *    Store result in 
							 */

							jointProb = (p1p2Count + m_Weight/(m_NumClasses * m_NumAttValues[p1] * m_NumAttValues[p2])) / ((m_SumInstances - missingForP1orP2) + m_Weight);							
							m_2vCondiCountsJoint[(m_2vOffsets[p1val] + p2val) * m_NumClasses + classval] = jointProb;

						} // ends classval

					} // ends p2val
				} // ends p2

			} // ends p1val
		} // ends p1

		/*
		 * 
		 *  
		 */
		for (int p1 = m_NumAttributes - 1; p1 >= 0; p1--) { // parent 1
			if (p1 == m_ClassIndex) continue;

			double missingForP1 = 0;
			missingForP1 = m_Frequencies[m_StartAttIndex[p1] + m_NumAttValues[p1]];

			for (int p1val = m_StartAttIndex[p1]; 
			p1val < m_StartAttIndex[p1] + m_NumAttValues[p1]; p1val++) {	
				for (int classval = 0; classval < m_NumClasses; classval++) {

					p1Count =  getCountFromTable(classval,p1val,p1val,p1val); 

					/* 
					 * Compute P(xp1,y)
					 *  
					 *              C(xp1,xp1,xp1) + m/(Card(xp1)*Card(y))
					 * P(xp1,y) = --------------------------------------
					 *                   (N - ?(xp1,y)) + m
					 * 
					 *    Store result in m_CondiCounts
					 */
					jointProb = (p1Count + m_Weight/(m_NumClasses * m_NumAttValues[p1])) / ((m_SumInstances - missingForP1) + m_Weight);
					m_1vCondiCounts[(p1val * m_NumClasses) + classval] = jointProb;

					/* 
					 * Compute P(xp1|y)
					 * 
					 *                  C(xp1,xp1,xp1,y) + m/|xp1|
					 * 		P(xp1|y) = -----------------------------
					 *                      (#N - ?xp)  + m 
					 */
					condProb = (p1Count + m_Weight/m_NumAttValues[p1]) / (m_SumForCounts[classval][p1] + m_Weight);
					m_1vCondiCountsNB[(p1val * m_NumClasses) + classval] = condProb;
				}
			}

		}

		/*
		 * Convert class counts into probabilities that is compute P(y)
		 */
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			m_ClassProbabilities[classVal] = MEsti(m_ClassCounts[classVal], m_SumInstances, m_NumClasses);
		}


	}

	/**
	 * Updates the classifier with the given instance.
	 *
	 * @param instance the new training instance to include in the model 
	 * @exception Exception if the instance could not be incorporated in
	 * the model.
	 */
	public void updateClassifier(Instance instance) {

		if (m_UseDiscretization) {
			m_Disc.input(instance);
			instance = m_Disc.output();
		}

		if (m_Incremental)
			this.addToCounts(instance);
		else {
			System.err.println("Classifier is not incremental");			
		}
	}

	/** Puts an instance's values into m_CondiCounts, m_ClassCounts and 
	 * m_SumInstances.
	 *
	 * @param instance the instance whose values are to be put into counts
	 *
	 */
	private void addToCounts(Instance instance) {

		if(instance.classIsMissing())
			return;   // ignore instances with missing class

		int classVal = (int)instance.classValue();
		double weight = instance.weight();

		m_ClassCounts[classVal] += weight;
		m_SumInstances += weight;

		// store instance's attribute value indexes in an array, b/c accessing it 
		// in loop(s) is more efficient
		int [] attIndex = new int[m_NumAttributes];
		for(int i = 0; i < m_NumAttributes; i++) {
			if(i == m_ClassIndex)
				attIndex[i] = -1;  // we don't use the class attribute in counts
			else {
				if(instance.isMissing(i))
					attIndex[i] = m_StartAttIndex[i] + m_NumAttValues[i];
				else {
					attIndex[i] = m_StartAttIndex[i] + (int)instance.value(i);
					m_SumForCounts[classVal][i] += weight;
				}
			}
		}

		for (int Att1 = m_NumAttributes - 1; Att1 >= 0; Att1--) {
			int Att1Index = attIndex[Att1];
			if (attIndex[Att1] == -1)
				continue;   // avoid pointless looping as Att1 is class attribute

			m_Frequencies[Att1Index] += weight;
			addCountinTable(classVal, Att1Index, Att1Index, Att1Index, weight);

			for (int Att2 = Att1 - 1; Att2 >= 0; Att2--) {
				int Att2Index = attIndex[Att2];
				if (attIndex[Att2] != -1) {

					m_Frequencies2p[attIndex[Att1]][attIndex[Att2]] += weight;
					addCountinTable(classVal, Att1Index, Att2Index, Att2Index, weight);

					for(int Att3 = Att2 - 1; Att3 >= 0; Att3--) {
						int Att3Index = attIndex[Att3];
						if (attIndex[Att3] != -1) {
							addCountinTable(classVal, Att1Index, Att2Index, Att3Index, weight);
						}
					} //ends Att3
				}
			} // ends Att2
		} // end Att1
	}

	/**
	 * Add count in m_1vCondiCount, m_2vCondiCount or m_3vCondiCount.
	 * 
	 * @param the class value
	 * @param parent 1 index
	 * @param parent 2 index
	 * @param child index
	 * @param weight
	 */
	public void addCountinTable(int classVal, int att1Index, int att2Index, int att3Index, double weight) {
		if (att1Index == att2Index && att2Index == att3Index)
			m_1vCondiCounts[(att1Index * m_NumClasses) + classVal] += weight;
		else if (att2Index == att3Index) {
			m_2vCondiCounts[(m_2vOffsets[att1Index] + att2Index) * m_NumClasses + classVal] += weight;
		} else {
			m_3vCondiCounts[(m_3vOffsets[att1Index] + m_2vOffsets[att2Index] + att3Index) * m_NumClasses + classVal] += weight;
		}
	}

	/**
	 * Get count from m_1vCondiCount, m_2vCondiCount or m_3vCondiCount.
	 * 
	 * @param the class value
	 * @param parent 1 index
	 * @param parent 2 index
	 * @param child index
	 * @return return count in m_1vCondiCount, m_2vCondiCount or m_3vCondiCount. 
	 */
	public double getCountFromTable(int classVal, int att1Index, int att2Index, int att3Index) {
		if (att1Index == att2Index && att2Index == att3Index) {
			return m_1vCondiCounts[(att1Index * m_NumClasses) + classVal];
		} else if (att2Index == att3Index) {
			return m_2vCondiCounts[(m_2vOffsets[att1Index] + att2Index) * m_NumClasses + classVal];
		} else {
			return m_3vCondiCounts[(m_3vOffsets[att1Index] + m_2vOffsets[att2Index] + att3Index) * m_NumClasses + classVal];
		}
	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception if there is a problem generating the prediction
	 */
	@Override
	public double [] distributionForInstance(Instance instance) throws Exception {

		if (m_UseDiscretization) {
			m_Disc.input(instance);
			instance = m_Disc.output();
		}

		double [] probs = new double[m_NumClasses];
		int p1Index, p2Index, parentCount = 0, childIndex;

		double[][] spodeProbs = new double[m_NumClasses][m_NumAttributes*(m_NumAttributes/2)];
		double[][] classParentsFreq = new double[m_NumClasses][m_NumAttributes*(m_NumAttributes/2)];
		double parentFreq;
		int p1p2Index = 0, p1Base = 0, p2Base;

		int[] SpecialGeneralArray = new int[m_NumAttributes];
		double counts;

		// store instance's att values in an int array, so accessing them 
		// is more efficient in loop(s).
		int [] attIndex = new int[m_NumAttributes];
		for(int att = 0; att < m_NumAttributes; att++) {
			if(instance.isMissing(att) || att == m_ClassIndex)
				attIndex[att] = -1;   // can't use class or missing in calculations
			else
				attIndex[att] = m_StartAttIndex[att] + (int)instance.value(att);
		}

		// -1 indicates attribute is not a generalization of any other attributes
		for (int i = 0; i < m_NumAttributes; i++) {
			SpecialGeneralArray[i] = -1;
		}

		/* Do subsumption Resolution */
		if (m_SubsumptionResolution) {			

			// calculate the specialization-generalization array
			for (int i = 0; i < m_NumAttributes; i++) {
				// skip i if it's the class or is missing
				if (attIndex[i] == -1)  continue;

				for (int j = 0; j < m_NumAttributes; j++) {
					// skip j if it's the class, missing, is i or a generalization of i
					if ((attIndex[j] == -1) || (i == j) || (SpecialGeneralArray[j] == i))
						continue;

					if (i < j) {
						counts = m_Frequencies2p[attIndex[j]][attIndex[i]]; //m_2vCondiCountsNoClass[m_2vOffsets[attIndex[j]] + attIndex[i]]; 
					} else {
						counts = m_Frequencies2p[attIndex[i]][attIndex[j]]; //m_2vCondiCountsNoClass[m_2vOffsets[attIndex[i]] + attIndex[j]];
					}

					// check j's frequency is above critical value
					if (m_Frequencies[attIndex[j]] > m_Critical) {

						// skip j if the frequency of i and j together is not equivalent
						// to the frequency of j alone
						if (m_Frequencies[attIndex[j]] == counts) {

							// if attributes i and j are both a specialization of each other
							// avoid deleting both by skipping j
							if ((m_Frequencies[attIndex[j]] == m_Frequencies[attIndex[i]]) && 
									(i < j)) {
								continue;
							} else {
								// set the specialization relationship
								SpecialGeneralArray[i] = j;
								break; // break out of j loop because a specialization has been found
							}
						}
					}
				}
			}

		}

		// calculate the piror probabilities before hand.
		for (int p1 = 1; p1 < m_NumAttributes; p1++) {
			p1Base += p1;
			p1p2Index = p1Base;

			if (attIndex[p1] == -1) continue;			
			// delete the generalization attributes.
			if (SpecialGeneralArray[p1] != -1) continue;

			p1Index = attIndex[p1];			
			double missingForP1 = m_Frequencies[m_StartAttIndex[p1] + m_NumAttValues[p1]];

			for (int p2 = 0; p2 < p1; p2++, p1p2Index++) {
				if (attIndex[p2] == -1) continue;	
				// delete the generalization attributes.
				if (SpecialGeneralArray[p2] != -1) continue;

				p2Index = attIndex[p2];
				double missingForP2 = m_Frequencies[m_StartAttIndex[p2] + m_NumAttValues[p2]];

				if (m_Frequencies2p[p1Index][p2Index] >= m_Limit) {						
					parentCount++;
					for (int classVal = 0; classVal < m_NumClasses; classVal++) {
						if (m_Incremental) {
							/*
							 *                    C(xp1,xp2,y) + m/(|xp1|*|xp2|)
							 * 		P(xp1,xp2) = ---------------------------------
							 *                      (#N - ?(xp1,xp2))  + m 
							 * 
							 */
							parentFreq = getCountFromTable(classVal, p1Index, p2Index, p2Index);

							double missingForP1orP2  = missingForP1 + missingForP2 - m_Frequencies2p[m_StartAttIndex[p1] + m_NumAttValues[p1]][m_StartAttIndex[p2] + m_NumAttValues[p2]];
							spodeProbs[classVal][p1p2Index] = m_mutualInformation[p1][p2] * probInitializerA2DE * ((parentFreq + m_Weight/(m_NumClasses * m_NumAttValues[p1] * m_NumAttValues[p2])) / ((m_SumInstances - missingForP1orP2) + m_Weight));

							classParentsFreq[classVal][p1p2Index] = parentFreq;
						} else {
							double a = m_2vCondiCountsJoint[(m_2vOffsets[p1Index] + p2Index) * m_NumClasses + classVal];
							spodeProbs[classVal][p1p2Index] = m_mutualInformation[p1][p2] * probInitializerA2DE * m_2vCondiCountsJoint[(m_2vOffsets[p1Index] + p2Index) * m_NumClasses + classVal];								
						}
					} // ends class
				} else {
					for (int classVal = 0; classVal < m_NumClasses; classVal++) {
						parentFreq = getCountFromTable(classVal, p1Index, p2Index, p2Index);;
						classParentsFreq[classVal][p1p2Index] = parentFreq;
					} // ends class
				}
			} // ends p2
		} // ends p1

		if (parentCount < 1) {
			//System.out.println("Resorting to AODE");
			return AODEconditionalProb(instance, attIndex, SpecialGeneralArray);
		}

		double childCondProb = 0, p1CondProb = 0, p2CondProb = 0;
		double missingForParent1 = 0, missingForParent2 = 0, missingForChild = 0;		

		p1p2Index = 0; p1Base = 0; p2Base = 0;
		for (int p1 = 1;  p1 < m_NumAttributes; p1++) {
			p1Base += p1;
			p1p2Index = p1Base;

			if (attIndex[p1] == -1) continue;
			// delete the generalization attributes.
			if (SpecialGeneralArray[p1] != -1) continue;

			p1Index = attIndex[p1];			

			p2Base = 0;
			for (int p2 = 0; p2 < p1; p2++, p1p2Index ++) {
				p2Base += p2;

				if (attIndex[p2] == -1) continue;
				// delete the generalization attributes.
				if (SpecialGeneralArray[p2] != -1) continue;

				p2Index = attIndex[p2];			

				int p1AttIndex = p1Base; 
				int p2AttIndex = p2Base;

				for (int child = 0; child < p2; child++, p1AttIndex++, p2AttIndex++) {
					if (attIndex[child] == -1) continue;
					// delete the generalization attributes.
					if (SpecialGeneralArray[child] != -1) continue;

					childIndex = attIndex[child];					

					for (int classVal = 0; classVal < m_NumClasses; classVal++) {
						if (m_Incremental) {

							double p1p2cCount = getCountFromTable(classVal, p1Index, p2Index, childIndex);

							/* 
							 *                     C(xc,xp1,xp2,y) + m/Card(xc)
							 * P(xc|xp1,xp2,y) = --------------------------------
							 *                     (C(xp1,xp2,y) - ?(xc)) + m
							 * 
							 * where Card(xc) is cardinality of xc and 
							 *       ?(xc) are the number of missing values of xc
							 */
							missingForChild = m_3vCondiCounts[(m_3vOffsets[p1Index] + m_2vOffsets[p2Index] + m_StartAttIndex[child] + m_NumAttValues[child]) * m_NumClasses + classVal];

							childCondProb = (p1p2cCount + m_Weight/m_NumAttValues[child]) / ((classParentsFreq[classVal][p1p2Index] - missingForChild) + m_Weight);							
							spodeProbs[classVal][p1p2Index] *= childCondProb;


							/* 
							 *                     C(xc,xp1,xp2,y) + m/Card(xp1)
							 * P(xp1|xc,xp2,y) = --------------------------------
							 *                     (C(xc,xp2,y) - ?(xp1)) + m
							 * 
							 * where Card(xp1) is cardinality of xp1 and 
							 *       ?(xp1) are the number of missing values of xp1
							 */
							missingForParent1 = m_3vCondiCounts[(m_3vOffsets[m_StartAttIndex[p1] + m_NumAttValues[p1]] + m_2vOffsets[p2Index] + childIndex) * m_NumClasses + classVal];

							p1CondProb = (p1p2cCount + m_Weight/m_NumAttValues[p1]) / ((classParentsFreq[classVal][p2AttIndex] - missingForParent1) + m_Weight);							
							spodeProbs[classVal][p2AttIndex] *= p1CondProb;							


							/*
							 *                      C(xc,xp1,xp2,y) + m/Card(xp2)
							 * P(xp2|xc,xp1,y) = --------------------------------
							 *                     (C(xc,xp1,y) - ?(xp2)) + m
							 * 
							 * where Card(xp2) is cardinality of xc and 
							 *       ?(xp2) are the number of missing values of xc
							 */
							missingForParent2 = m_3vCondiCounts[(m_3vOffsets[p1Index] + m_2vOffsets[m_StartAttIndex[p2] + m_NumAttValues[p2]] + childIndex) * m_NumClasses + classVal];

							p2CondProb = (p1p2cCount + m_Weight/m_NumAttValues[p2]) / ((classParentsFreq[classVal][p1AttIndex] - missingForParent2) + m_Weight);							
							spodeProbs[classVal][p1AttIndex] *= p2CondProb;

						} else {
							spodeProbs[classVal][p1p2Index] *= m_3vCondiCounts[(m_3vOffsets[p1Index] + m_2vOffsets[p2Index] + childIndex) * m_NumClasses + classVal];
							spodeProbs[classVal][p2AttIndex] *= m_3vCondiCountsP1[(m_3vOffsets[p1Index] + m_2vOffsets[p2Index] + childIndex) * m_NumClasses + classVal];
							spodeProbs[classVal][p1AttIndex] *= m_3vCondiCountsP2[(m_3vOffsets[p1Index] + m_2vOffsets[p2Index] + childIndex) * m_NumClasses + classVal];
						}
					} // ends classVal	
				} // ends child
			} // ends p2
		} // ends p1

		/* add all the probabilities for each class */
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			for (int i = 0; i < spodeProbs[classVal].length; i++) {
				probs[classVal] += spodeProbs[classVal][i];
			}
		}

		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * 
	 * @param instance the instance to be classified
	 * @param attIndex attribute value indexes for the instance
	 * @return predicted class probability distribution
	 * @throws Exception if there is a problem generating the prediction
	 */
	public double[] AODEconditionalProb(Instance instance, int[] attIndex, int[] SpecialGeneralArray) throws Exception {

		// accumulates posterior probabilities for each class
		double[] probs = new double[m_NumClasses];		

		double[][] spodeProbs = new double[m_NumClasses][m_NumAttributes];
		double[][] classParentsFreq = new double[m_NumClasses][m_NumAttributes];
		double parentFreq;
		int parentCount = 0;
		int pIndex, childIndex;

		/* Get the prior probabilities first, each attribute has a turn of being the parent. */
		for (int parent = 0; parent < m_NumAttributes; parent++) {
			pIndex = attIndex[parent];
			if (pIndex == -1) continue;

			// delete the generalization attributes.
			if (SpecialGeneralArray[parent] != -1) 
				continue;

			if (m_Frequencies[pIndex] >= m_Limit) {
				parentCount++;
				double missing4ParentAtt = m_Frequencies[m_StartAttIndex[parent] + m_NumAttValues[parent]];

				// calculate the prior probability -- P(parent & classVal)
				for (int classVal = 0; classVal < m_NumClasses; classVal++) {					

					if (m_Incremental) {
						/* 
						 * Get the count C(x1,x1,y) from m_CondiCount and compute the joint
						 * probability as
						 * 
						 *             C(x1,x1,x1,y) + m/(Card(x1)*Card(y))
						 * P(x1,y) = ------------------------------------
						 *                   (N - ?(x1,y)) + m
						 *   
						 */	
						parentFreq =  getCountFromTable(classVal, pIndex, pIndex, pIndex); 
						spodeProbs[classVal][parent] = probInitializerAODE * MEsti(parentFreq, (m_SumInstances - missing4ParentAtt), (m_NumClasses * m_NumAttValues[parent]));		
						classParentsFreq[classVal][parent] = parentFreq;						
					} else {
						/* No need to compute probabilities from the counts.
						 * The probability P(x1,x1,y) is stored at the diagonal element
						 * of m_CondiCounts. So just retrieve the value */
						spodeProbs[classVal][parent] = probInitializerAODE * m_1vCondiCounts[(pIndex * m_NumClasses) + classVal];
					}
				}	
			} else {
				for (int classVal = 0; classVal < m_NumClasses; classVal++) {
					parentFreq =  getCountFromTable(classVal, pIndex, pIndex, pIndex);;
					classParentsFreq[classVal][parent] = parentFreq;
				}
			}
		}

		// check that at least one parent was used, else do NB
		if (parentCount < 1) {
			//System.out.println("Doing NB");
			return NBconditionalProb(instance, attIndex);
		}

		for (int parent = 1; parent < m_NumAttributes; parent++) {
			pIndex = attIndex[parent];
			if (pIndex == -1)
				continue;

			// delete the generalization attributes.
			if (SpecialGeneralArray[parent] != -1) 
				continue;

			for (int child = 0; child < parent; child++) {
				childIndex = attIndex[child];
				if (childIndex == -1)
					continue;

				// delete the generalization attributes.
				if (SpecialGeneralArray[parent] != -1) 
					continue;

				double missingForParent = 0, missingForChild = 0, countForAttOpp = 0, countForAtt = 0;

				for (int classVal = 0; classVal < m_NumClasses; classVal++) {

					if (m_Incremental) {
						// Get the count C(xp,xc,xc) from m_CondiCounts table
						countForAtt = getCountFromTable(classVal, pIndex, childIndex, childIndex);

						/* 
						 * Compute probability
						 * 
						 *              C(xp,xc,xc) + m/|xp|
						 * P(xp|xc) = ------------------------
						 *              {#(xc) - ?xp} + m
						 */
						missingForParent = m_2vCondiCounts[(m_2vOffsets[m_StartAttIndex[parent] + m_NumAttValues[parent]] + childIndex) * m_NumClasses + classVal];
						spodeProbs[classVal][child] *= (countForAtt + m_Weight/m_NumAttValues[parent]) / ((classParentsFreq[classVal][child] - missingForParent) + m_Weight);						

						/* 
						 * Compute probability
						 *  
						 *              C(xp,xc,xc) + m/|xc|
						 *  P(xc|xp) = -----------------------
						 *              {C(xp) - ?xc} + m
						 */
						missingForChild = m_2vCondiCounts[(m_2vOffsets[pIndex] + (m_StartAttIndex[child] + m_NumAttValues[child])) * m_NumClasses + classVal];
						spodeProbs[classVal][parent] *= (countForAtt + m_Weight/m_NumAttValues[child]) / ((classParentsFreq[classVal][parent] - missingForChild) + m_Weight);
					} else {
						spodeProbs[classVal][child] *= m_2vCondiCounts[(m_2vOffsets[pIndex] + childIndex) * m_NumClasses + classVal]; 
						spodeProbs[classVal][parent] *= m_2vCondiCountsOpp[(m_2vOffsets[pIndex] + childIndex) * m_NumClasses + classVal];						
					}

				} // ends classVal
			} // ends child
		} // ends parent

		/* add all the probabilities for each class */
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			for (int i = 0; i < m_NumAttributes; i++) {
				probs[classVal] += spodeProbs[classVal][i];
			}
		}

		Utils.normalize(probs);
		return probs;
	}

	/**
	 * Calculates the probability of the specified class for the given test
	 * instance, using naive Bayes.
	 *
	 * @param instance the instance to be classified
	 * @param classVal the class for which to calculate the probability
	 * @return predicted class probability
	 * @throws Exception 
	 */
	public double[] NBconditionalProb(Instance instance, int[] attIndex) throws Exception {

		double[] probs = new double[m_NumClasses];
		double countForAtt;

		// calculate the prior probability
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			if (m_Incremental) {
				probs[classVal] = probInitializer * MEsti(m_ClassCounts[classVal], m_SumInstances, m_NumClasses);
			} else {
				// No need to compute the probability
				probs[classVal] = probInitializer * m_ClassProbabilities[classVal];	
			}
		}

		// consider effect of each att value
		for (int child = 0; child < m_NumAttributes; child++) {
			// determine correct index for att in m_CondiCounts
			int childIndex = attIndex[child];
			if (attIndex[child] == -1)
				continue;

			for (int classVal = 0; classVal < m_NumClasses; classVal++) {
				if (m_Incremental) {
					/* Get the count C(x1,x1,x1,y) from m_CondiCount and compute the conditional
					 * probability as
					 * 
					 *                  C(x1,x1,x1,y) + m/|x1|
					 * 		P(x1|y) = --------------------------
					 *                  (#N - ?x1)  + m 
					 */
					countForAtt = getCountFromTable(classVal, childIndex, childIndex, childIndex); 
					probs[classVal] *= MEsti(countForAtt, m_SumForCounts[classVal][child], m_NumAttValues[child]); 
				} else {
					/* No need to compute conditional probability from the counts, 
					 * This is stored at m_1vCondiCountsNB, so just retrieve the value
					 */
					probs[classVal] *= m_1vCondiCountsNB[(childIndex * m_NumClasses) + classVal];
				}
			}
		}

		Utils.normalize(probs);
		return probs;  
	}

	/**
	 * Performs m-estimation
	 */
	public double MEsti(double freq1, double freq2, double numValues) throws Exception {
		double mEsti = (freq1 + m_Weight / numValues) / (freq2 + m_Weight);
		return mEsti;
	}

	/**
	 * Returns an enumeration describing the available options
	 *
	 * @return an enumeration of all the available options
	 */
	@Override
	public Enumeration listOptions() {

		Vector newVector = new Vector(4);

		newVector.addElement(new Option("\tOutput debugging information\n", "D", 0,"-D"));
		newVector.addElement(new Option("\tImpose a frequency limit for superParents \t (default is 1)", "F", 1,"-F <int>"));
		newVector.addElement( new Option("\tSpecify a weight to use with m-estimate (default is 1) \n", "M", 1, "-M <double>"));
		newVector.addElement( new Option("\tSpecify a critical value for specialization-generalilzation SR (default is 100) \n", "S", 1, "-S <int>"));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -F &lt;int&gt;
	 *  Impose a frequency limit for superParents
	 *  (default is 1)</pre>
	 * 
	 * <pre> -M &lt;int&gt;
	 *  Specify a weight to use with m-estimate
	 *  (default is 1)</pre>
	 *  
	 * <pre> -S (Optional) &lt;int&gt;
	 *  Specify critical value of specialization-generaliztion for 
	 *  subsumption resolution
	 *  (default is 100)
	 *  Results in lowering bias and increasing variance
	 *  
	 * <pre> -W (Optional) &lt;int&gt;
	 *  Do Weighted A2DE
	 *  Results in lowering bias and increasing variance
	 *  
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String Freq = Utils.getOption('F', options);
		if (Freq.length() != 0) 
			m_Limit = Integer.parseInt(Freq);
		else
			m_Limit = 1;

		String Weight = Utils.getOption('M', options);
		if (Weight.length() != 0)
			m_Weight = Double.parseDouble(Weight);			   
		else 
			m_Weight = 1;
		
		String m_SROption = Utils.getOption('S', options);		
		if (m_SROption.length() != 0) {
			m_SubsumptionResolution = true;
			m_Critical = Integer.parseInt(m_SROption);					
		} else { 
			m_Critical = 100;
		}
		
		m_WeightedA2DE = Utils.getFlag('W', options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String [] getOptions() {
		Vector result  = new Vector();

		result.add("-F");
		result.add("" + m_Limit);

		result.add("-M");
		result.add("" + m_Weight);
		
		if (m_SubsumptionResolution)    
			result.add("-S");
		
		if (m_WeightedA2DE)    
			result.add("-W");

		return (String[]) result.toArray(new String[result.size()]);
	}

	/**
	 * Sets the frequency limit
	 *
	 * @param f the frequency limit
	 */
	public void setFrequencyLimit(int f) {
		m_Limit = f;
	}

	/**
	 * Gets the frequency limit.
	 *
	 * @return the frequency limit
	 */
	public int getFrequencyLimit() {
		return m_Limit;
	}

	/**
	 * Sets the weight for m-estimate
	 *
	 * @param w the weight
	 */
	public void setWeight(double weight) {
		if (weight > 0)
			m_Weight = weight;
		else
			System.out.println("Weight must be greater than 0!");
	}

	/**
	 * Gets the weight used in m-estimate
	 *
	 * @return the frequency limit
	 */
	public double getWeight() {
		return m_Weight;
	}
	
	/**
	 * Sets the Subsumption Resolution flag
	 *
	 * @param S the Subsumption Resolution flag
	 */
	public void setSubsumptionResolution(boolean S) {
		m_SubsumptionResolution = S;
	}

	/**
	 * Gets the Subsumption Resolution flag.
	 *
	 * @return the SubsumptionResolution flag
	 */
	public boolean getSubsumptionResolution() {
		return m_SubsumptionResolution;
	}
	
	/**
	 * Sets the Weighted A2DE flag
	 *
	 * @param f the Weighted A2DE flag
	 */
	public void setWeightedA2DE(boolean W) {
		m_WeightedA2DE = W;
	}

	/**
	 * Gets the Weighted A2DE flag.
	 *
	 * @return the Weighted A2DE flag
	 */
	public boolean getWeightedA2DE() {
		return m_WeightedA2DE;
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {

		StringBuffer text = new StringBuffer();
		text.append("The A2DE Classifier\n");
		if (m_Instances == null) {
			text.append(": No model built yet.");
		} else {
			try {				
				for (int i = 0; i < m_NumClasses; i++)
					text.append("\nClass " + m_Instances.classAttribute().value(i) + 
							": Prior probability = " + 
							Utils.doubleToString(((m_ClassCounts[i] + 1) / 
									(m_SumInstances + m_NumClasses)), 4, 2));

				text.append("\n\nDataset: " + m_Instances.relationName() + "\n" + 
						"Instances: " + m_NumInstances + "\n" + "Attributes: " + m_NumAttributes + 
						"\n" + "Frequency limit for superParents: (F = " + m_Limit + ") \n");
				text.append("Correction: ");
				text.append("m-estimate (m = " + m_Weight + ")\n");
				text.append("Incremental Classifier Flag: (" + m_Incremental + ")\n");
				text.append("Subsumption Resolution Flag: (" + m_SubsumptionResolution + ")\n");
				text.append("Critical Value for Subsumption Resolution (" + m_Critical + ")\n");
				text.append("Weighted A2DE Flag: (" + m_WeightedA2DE + ")\n");
			} catch (Exception ex) {
				text.append(ex.getMessage());
			}
		}

		return text.toString();
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param args the options
	 */
	public static void main(String[] args) {
		runClassifier(new A2DE(), args);
	}

}
