#!/usr/bin/env python3
"""
Waze User Retention Prediction Script

A production-ready script that takes user profile data and returns churn risk predictions.
This script loads the trained model and provides clean API for churn prediction.

Usage:
    python predict.py --user_data "user_profile.json"
    python predict.py --help

Author: Data Science Team
Date: 2026-02-16
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

class WazeChurnPredictor:
    """
    Production churn prediction system for Waze users.
    """
    
    def __init__(self, model_path='../models/waze_retention_model.pkl'):
        """
        Initialize the predictor with trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model_package = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and associated components."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model: {self.model_package['model_name']}")
            print(f"🎯 F1-Score: {self.model_package['model_performance']['F1-Score']:.3f}")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at {self.model_path}")
            print("   Please run the training notebook first to generate the model.")
            sys.exit(1)
    
    def engineer_features(self, user_data):
        """
        Apply the same feature engineering as training.
        
        Args:
            user_data (dict): Raw user profile data
            
        Returns:
            dict: Engineered features
        """
        # Avoid division by zero
        epsilon = 0.001
        
        # Create engineered features
        user_data['km_per_driving_day'] = user_data['driven_km_drives'] / (user_data['driving_days'] + epsilon)
        user_data['sessions_per_day'] = user_data['sessions'] / (user_data['activity_days'] + epsilon)
        user_data['drives_per_session'] = user_data['drives'] / (user_data['sessions'] + epsilon)
        user_data['km_per_drive'] = user_data['driven_km_drives'] / (user_data['drives'] + epsilon)
        user_data['minutes_per_drive'] = user_data['duration_minutes_drives'] / (user_data['drives'] + epsilon)
        user_data['activity_consistency'] = user_data['driving_days'] / (user_data['activity_days'] + epsilon)
        user_data['navigation_intensity'] = (user_data['total_navigations_fav1'] + user_data['total_navigations_fav2']) / (user_data['sessions'] + epsilon)
        
        # Additional features
        user_data['total_navigation'] = user_data['total_navigations_fav1'] + user_data['total_navigations_fav2']
        user_data['engagement_score'] = (user_data['sessions'] + user_data['drives'] + user_data['total_navigation']) / 3
        user_data['km_per_drive'] = user_data['driven_km_drives'] / (user_data['drives'] + epsilon)
        user_data['efficiency_ratio'] = user_data['driven_km_drives'] / (user_data['driving_days'] + epsilon)
        
        return user_data
    
    def predict_churn_risk(self, user_profile):
        """
        Predict churn risk for a single user.
        
        Args:
            user_profile (dict): User profile data
            
        Returns:
            dict: Prediction results with risk score and recommendation
        """
        # Apply feature engineering
        engineered_profile = self.engineer_features(user_profile.copy())
        
        # Create feature vector in correct order
        feature_columns = self.model_package['feature_columns']
        feature_values = []
        
        for feature in feature_columns:
            if feature in engineered_profile:
                feature_values.append(engineered_profile[feature])
            else:
                # Handle missing features with defaults
                feature_values.append(0)
        
        # Convert to numpy array and predict
        X = np.array(feature_values).reshape(1, -1)
        model = self.model_package['model']
        
        # Get prediction and probability
        churn_prediction = model.predict(X)[0]
        churn_probability = model.predict_proba(X)[0][1]  # Probability of churn
        
        # Generate risk assessment
        risk_level = self.assess_risk_level(churn_probability)
        
        return {
            'user_id': user_profile.get('ID', 'Unknown'),
            'churn_prediction': bool(churn_prediction),
            'churn_probability': float(churn_probability),
            'risk_level': risk_level,
            'recommendation': self.get_recommendation(churn_probability, risk_level),
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': self.model_package['model_name']
        }
    
    def assess_risk_level(self, probability):
        """Assess risk level based on churn probability."""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_recommendation(self, probability, risk_level):
        """Generate actionable recommendations based on risk level."""
        if risk_level == "HIGH":
            return "URGENT: Immediate intervention needed. Contact user with retention offer."
        elif risk_level == "MEDIUM":
            return "MONITOR: Send engagement campaign. Check usage patterns weekly."
        else:
            return "MAINTAIN: User is stable. Continue standard service."
    
    def predict_batch(self, user_profiles_list):
        """
        Predict churn risk for multiple users.
        
        Args:
            user_profiles_list (list): List of user profile dictionaries
            
        Returns:
            list: List of prediction results
        """
        results = []
        for i, profile in enumerate(user_profiles_list):
            try:
                result = self.predict_churn_risk(profile)
                results.append(result)
                print(f"✅ Processed user {i+1}/{len(user_profiles_list)}")
            except Exception as e:
                print(f"❌ Error processing user {i+1}: {str(e)}")
                results.append({
                    'error': str(e),
                    'user_id': profile.get('ID', f'user_{i+1}')
                })
        
        return results

def load_user_data(file_path):
    """Load user data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in file - {file_path}")
        return None

def create_sample_user():
    """Create a sample user profile for testing."""
    return {
        "ID": "sample_user_001",
        "sessions": 150,
        "drives": 120,
        "total_sessions": 180.5,
        "n_days_after_onboarding": 1500,
        "total_navigations_fav1": 50,
        "total_navigations_fav2": 10,
        "driven_km_drives": 2500,
        "duration_minutes_drives": 1200,
        "activity_days": 25,
        "driving_days": 18,
        "device_iPhone": 1  # 1 for iPhone, 0 for Android
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Waze User Churn Prediction System')
    parser.add_argument('--user_data', type=str, 
                       help='Path to JSON file containing user profile data')
    parser.add_argument('--sample', action='store_true',
                       help='Run prediction on sample user data')
    parser.add_argument('--model_path', type=str, 
                       default='../models/waze_retention_model.pkl',
                       help='Path to the trained model file')
    
    args = parser.parse_args()
    
    print("🚀 Waze User Retention Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = WazeChurnPredictor(args.model_path)
    
    if args.sample:
        # Use sample data
        print("📊 Using sample user data...")
        user_data = create_sample_user()
        result = predictor.predict_churn_risk(user_data)
        
        print(f"\n🎯 PREDICTION RESULTS")
        print("=" * 30)
        print(f"User ID: {result['user_id']}")
        print(f"Churn Risk: {result['churn_probability']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Will Churn: {'Yes' if result['churn_prediction'] else 'No'}")
        print(f"Recommendation: {result['recommendation']}")
        
    elif args.user_data:
        # Load user data from file
        user_data = load_user_data(args.user_data)
        if user_data is None:
            return
        
        # Handle single user or batch
        if isinstance(user_data, dict):
            # Single user
            result = predictor.predict_churn_risk(user_data)
            print(f"\n🎯 PREDICTION RESULTS")
            print("=" * 30)
            print(json.dumps(result, indent=2))
        elif isinstance(user_data, list):
            # Multiple users
            results = predictor.predict_batch(user_data)
            print(f"\n🎯 BATCH PREDICTION RESULTS")
            print("=" * 30)
            for result in results:
                print(json.dumps(result, indent=2))
                print("-" * 20)
    else:
        print("❌ Please provide user data with --user_data or use --sample for testing")
        print("Example: python predict.py --sample")
        return

if __name__ == "__main__":
    main()