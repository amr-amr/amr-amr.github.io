---
layout: archive
title: "Literature Review"
permalink: /litreview/
author_profile: false
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.litreview reversed %}
  {% include archive-single.html %}
{% endfor %}
