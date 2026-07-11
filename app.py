from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Product, Interaction, Order, OrderItem
from recommender import ProductRecommender
import json
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecommerce.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize recommender
recommender = ProductRecommender()

# Load product data
def load_products():
    with open('data/products.json', 'r') as f:
        products_data = json.load(f)
    return products_data

# Create tables and load data
with app.app_context():
    db.create_all()
    
    # Load products if empty
    if Product.query.count() == 0:
        products_data = load_products()
        for p in products_data:
            product = Product(
                id=p['id'],
                name=p['name'],
                category=p['category'],
                price=p['price'],
                description=p['description'],
                rating=p['rating'],
                tags=','.join(p['tags'])
            )
            db.session.add(product)
        db.session.commit()
    
    # Initialize recommender with products
    products = Product.query.all()
    recommender.load_products([p.to_dict() for p in products])
    
    # Load interactions for collaborative filtering
    interactions = Interaction.query.all()
    if interactions:
        interaction_data = [{'user_id': i.user_id, 'product_id': i.product_id, 'rating': i.rating or 3} 
                           for i in interactions]
        recommender.train_collaborative_model(interaction_data)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.context_processor
def inject_user():
    from flask_login import current_user
    return dict(current_user=current_user)

# Routes
@app.route('/')
def home():
    # Get recommendations for current user if logged in
    recommendations = []
    if current_user.is_authenticated:
        # Get user's interactions
        user_interactions = Interaction.query.filter_by(user_id=current_user.id).all()
        if user_interactions:
            history = [{'product_id': i.product_id} for i in user_interactions]
            recommendations = recommender.get_recommendations_for_user_history(history, top_n=6)
        else:
            recommendations = recommender.get_popular_products(6)
    else:
        recommendations = recommender.get_popular_products(6)
    
    # Get all products for display
    products = Product.query.all()
    return render_template('index.html', products=products, recommendations=recommendations)

@app.route('/products')
def products_page():
    category = request.args.get('category')
    search = request.args.get('search')
    
    query = Product.query
    if category:
        query = query.filter_by(category=category)
    if search:
        query = query.filter(Product.name.contains(search) | Product.description.contains(search))
    
    products = query.all()
    return render_template('products.html', products=products)

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    
    # Record view interaction
    if current_user.is_authenticated:
        interaction = Interaction(
            user_id=current_user.id,
            product_id=product_id,
            action_type='view'
        )
        db.session.add(interaction)
        db.session.commit()
        
        # Get recommendations
        recommendations = recommender.get_hybrid_recommendations(
            user_id=current_user.id,
            product_id=product_id,
            top_n=5
        )
    else:
        recommendations = recommender.get_content_based_recommendations(product_id, top_n=5)
    
    return render_template('product_detail.html', product=product, recommendations=recommendations)

@app.route('/cart')
@login_required
def cart():
    # Get cart items from session
    cart_items = session.get('cart', [])
    products = []
    total = 0
    for item in cart_items:
        product = Product.query.get(item['product_id'])
        if product:
            products.append({
                'product': product,
                'quantity': item['quantity']
            })
            total += product.price * item['quantity']
    
    return render_template('cart.html', cart_items=products, total=total)

@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
@login_required
def add_to_cart(product_id):
    cart = session.get('cart', [])
    
    # Check if product already in cart
    for item in cart:
        if item['product_id'] == product_id:
            item['quantity'] += 1
            break
    else:
        cart.append({'product_id': product_id, 'quantity': 1})
    
    session['cart'] = cart
    
    # Record interaction
    interaction = Interaction(
        user_id=current_user.id,
        product_id=product_id,
        action_type='cart'
    )
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify({'success': True, 'cart_count': len(cart)})

@app.route('/remove_from_cart/<int:product_id>', methods=['POST'])
@login_required
def remove_from_cart(product_id):
    cart = session.get('cart', [])
    cart = [item for item in cart if item['product_id'] != product_id]
    session['cart'] = cart
    return jsonify({'success': True})

@app.route('/checkout', methods=['POST'])
@login_required
def checkout():
    cart_items = session.get('cart', [])
    if not cart_items:
        return jsonify({'error': 'Cart is empty'}), 400
    
    # Create order
    total = 0
    order = Order(
        user_id=current_user.id,
        total_amount=0,
        status='completed'
    )
    db.session.add(order)
    db.session.flush()
    
    # Create order items
    for item in cart_items:
        product = Product.query.get(item['product_id'])
        if product:
            order_item = OrderItem(
                order_id=order.id,
                product_id=product.id,
                quantity=item['quantity'],
                price=product.price
            )
            db.session.add(order_item)
            total += product.price * item['quantity']
            
            # Record purchase interaction
            interaction = Interaction(
                user_id=current_user.id,
                product_id=product.id,
                action_type='purchase',
                rating=4  # Default rating
            )
            db.session.add(interaction)
    
    order.total_amount = total
    db.session.commit()
    
    # Clear cart
    session['cart'] = []
    
    # Retrain collaborative model with new interactions
    interactions = Interaction.query.all()
    if interactions:
        interaction_data = [{'user_id': i.user_id, 'product_id': i.product_id, 'rating': i.rating or 3} 
                           for i in interactions]
        recommender.train_collaborative_model(interaction_data)
    
    return jsonify({'success': True, 'order_id': order.id})

@app.route('/orders')
@login_required
def orders():
    user_orders = Order.query.filter_by(user_id=current_user.id).order_by(Order.created_at.desc()).all()
    return render_template('orders.html', orders=user_orders)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/rate_product/<int:product_id>', methods=['POST'])
@login_required
def rate_product(product_id):
    rating = request.json.get('rating')
    if not rating or not (1 <= rating <= 5):
        return jsonify({'error': 'Invalid rating'}), 400
    
    # Check if interaction exists
    interaction = Interaction.query.filter_by(
        user_id=current_user.id,
        product_id=product_id
    ).first()
    
    if interaction:
        interaction.rating = rating
    else:
        interaction = Interaction(
            user_id=current_user.id,
            product_id=product_id,
            action_type='rating',
            rating=rating
        )
        db.session.add(interaction)
    
    db.session.commit()
    
    # Update product rating
    product = Product.query.get(product_id)
    ratings = Interaction.query.filter_by(product_id=product_id).filter(Interaction.rating.isnot(None)).all()
    if ratings:
        avg_rating = sum(r.rating for r in ratings) / len(ratings)
        product.rating = round(avg_rating, 1)
        db.session.commit()
    
    # Retrain model
    interactions = Interaction.query.all()
    if interactions:
        interaction_data = [{'user_id': i.user_id, 'product_id': i.product_id, 'rating': i.rating or 3} 
                           for i in interactions]
        recommender.train_collaborative_model(interaction_data)
    
    return jsonify({'success': True, 'avg_rating': product.rating})

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for recommendations"""
    product_id = request.args.get('product_id', type=int)
    
    if current_user.is_authenticated:
        if product_id:
            recs = recommender.get_hybrid_recommendations(
                user_id=current_user.id,
                product_id=product_id,
                top_n=10
            )
        else:
            # Get user history
            user_interactions = Interaction.query.filter_by(user_id=current_user.id).all()
            if user_interactions:
                history = [{'product_id': i.product_id} for i in user_interactions]
                recs = recommender.get_recommendations_for_user_history(history, top_n=10)
            else:
                recs = recommender.get_popular_products(10)
    else:
        if product_id:
            recs = recommender.get_content_based_recommendations(product_id, top_n=10)
        else:
            recs = recommender.get_popular_products(10)
    
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True)