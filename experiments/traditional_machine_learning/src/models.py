import pickle
import sklearn.ensemble
import sklearn.svm
import sklearn.linear_model
import xgboost
from sklearn.pipeline import make_pipeline

class Model:

  @classmethod
  def create( cls, x, y, scale=False, **args ):
    model = cls()
    model.clf = model.build( **args )
    if scale == 'standardize':
        print('Use StandardScaler')
        scaler = sklearn.preprocessing.StandardScaler()
        model.clf = make_pipeline(scaler, model.clf)
    elif scale == 'minmax':
        print('Use MinMaxScaler')
        scaler = sklearn.preprocessing.MinMaxScaler()
        model.clf = make_pipeline(scaler, model.clf)
    model.train( x, y )
    return model 

  @classmethod
  def restore( cls, path ):
    """
    Restore from pickle.
    """
    model = cls()
    with open( path, 'rb' ) as f:
      model.clf = pickle.load( f )
    return model 

  def save( self, path ):
    """
    Dump to pickle.
    """
    with open( path, 'wb' ) as f:
      pickle.dump( self.clf, f )

  @property
  def Algorithm( self ):
    raise NotImplementedError

  def build( self, **args ):
    return self.Algorithm( **args )

  def train( self, x, y  ):
    self.clf.fit( x, y )

  def predict( self, x ):
    y_hat = self.clf.predict_proba( x )
    confidences = y_hat.max( axis=1 )
    predictions = self.clf.classes_[ y_hat.argmax( axis=1 ) ]
    return predictions, confidences


class XGBoostModel( Model ):
  # See: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
  Algorithm = xgboost.XGBClassifier

  # def train( self, x, y ):
  #   self.clf.fit( x, y, eval_set=[(x,y)] )

class RFModel( Model ):
  # See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  Algorithm = sklearn.ensemble.RandomForestClassifier

class SVMModel( Model ):
  # See https://scikit-learn.org/stable/modules/svm.html
  Algorithm = sklearn.svm.SVC 

class KNNModel( Model ):
    Algorithm = sklearn.neighbors.KNeighborsClassifier 

class LogisticModel:
  # See: logistic regression sklearn coefficients
  Algorithm = sklearn.linear_model.LogisticRegression

  
def get_model( algorithm ):
    if algorithm == 'xgboost':
      return XGBoostModel
    elif algorithm == 'rf':
      return RFModel
    elif algorithm == 'svm':
      return SVMModel
    elif algorithm == 'knn':
      return KNNModel
    elif algorithm == 'logistic':
      return LogisticModel
    else:
      raise ValueError( f'No algorithm with name "{algorithm}"' )
